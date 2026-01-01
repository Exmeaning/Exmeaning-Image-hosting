// scripts/process-images.mjs
import { AutoModel, RawImage, Tensor } from '@huggingface/transformers';
import sharp from 'sharp';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT_DIR = path.join(__dirname, '..');
const INPUT_DIR = path.join(ROOT_DIR, 'costume_origin');
const OUTPUT_DIR = path.join(ROOT_DIR, 'costume_icons');
const MAX_SIZE = 512;
const MODEL_INPUT_SIZE = 1024;

const HOURS = parseInt(process.env.HOURS || process.argv[2] || '168', 10);

const SUPPORTED_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.webp'];

/**
 * 获取指定小时数内修改的文件
 */
async function getRecentFiles(dir, hours) {
  const files = [];
  const cutoffTime = Date.now() - hours * 60 * 60 * 1000;

  try {
    const entries = await fs.readdir(dir, { withFileTypes: true });
    
    for (const entry of entries) {
      if (!entry.isFile()) continue;
      
      const ext = path.extname(entry.name).toLowerCase();
      if (!SUPPORTED_EXTENSIONS.includes(ext)) continue;
      
      const filePath = path.join(dir, entry.name);
      const stats = await fs.stat(filePath);
      
      if (stats.mtimeMs >= cutoffTime) {
        files.push({
          path: filePath,
          name: entry.name,
          mtime: stats.mtime
        });
      }
    }
  } catch (error) {
    if (error.code === 'ENOENT') {
      console.log(`📁 目录不存在: ${dir}`);
      return [];
    }
    throw error;
  }

  return files;
}

/**
 * 手动预处理图片（替代 AutoProcessor）
 */
async function preprocessImage(imagePath) {
  // 读取图片并转换为 RGB
  const { data, info } = await sharp(imagePath)
    .resize(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, { fit: 'fill' })
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });

  const { width, height } = info;
  const pixels = width * height;

  // 归一化参数 (ImageNet 标准)
  const mean = [0.485, 0.456, 0.406];
  const std = [0.229, 0.224, 0.225];

  // 创建 Float32Array 存储归一化后的数据 [1, 3, H, W]
  const float32Data = new Float32Array(3 * pixels);

  for (let i = 0; i < pixels; i++) {
    // RGB 归一化到 0-1，然后标准化
    float32Data[i] = (data[i * 3] / 255 - mean[0]) / std[0];                    // R
    float32Data[pixels + i] = (data[i * 3 + 1] / 255 - mean[1]) / std[1];       // G
    float32Data[2 * pixels + i] = (data[i * 3 + 2] / 255 - mean[2]) / std[2];   // B
  }

  return new Tensor('float32', float32Data, [1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE]);
}

/**
 * 使用 IS-Net-Anime 模型进行抠图
 */
async function removeBackground(imagePath, model) {
  console.log(`  🔄 加载图片...`);
  
  // 获取原始尺寸
  const metadata = await sharp(imagePath).metadata();
  const originalWidth = metadata.width;
  const originalHeight = metadata.height;
  
  console.log(`  📐 原始尺寸: ${originalWidth}x${originalHeight}`);
  
  // 预处理图片
  console.log(`  🔧 预处理图片...`);
  const inputTensor = await preprocessImage(imagePath);
  
  // 运行模型
  console.log(`  🤖 运行抠图模型...`);
  const output = await model({ input: inputTensor });
  
  // 获取 mask - 输出可能是 output.output 或直接的 tensor
  let maskTensor = output.output || output;
  if (Array.isArray(maskTensor)) {
    maskTensor = maskTensor[0];
  }
  
  const maskData = maskTensor.data;
  const maskSize = MODEL_INPUT_SIZE;
  
  return {
    maskData,
    maskSize,
    originalWidth,
    originalHeight
  };
}

/**
 * 应用 mask 并处理图片
 */
async function applyMaskAndProcess(imagePath, maskInfo) {
  const { maskData, maskSize, originalWidth, originalHeight } = maskInfo;
  
  // 将 mask 数据转换为 0-255 范围
  const maskBuffer = Buffer.alloc(maskSize * maskSize);
  for (let i = 0; i < maskSize * maskSize; i++) {
    // mask 值可能是 0-1 的 float 或 0-255 的 uint8
    const val = maskData[i];
    maskBuffer[i] = val > 1 ? val : Math.round(val * 255);
  }
  
  // 将 mask 调整为原始图片大小
  const resizedMask = await sharp(maskBuffer, {
    raw: {
      width: maskSize,
      height: maskSize,
      channels: 1
    }
  })
    .resize(originalWidth, originalHeight, { fit: 'fill' })
    .raw()
    .toBuffer();
  
  // 读取原始图片并转换为 RGBA
  const { data: rgbData, info } = await sharp(imagePath)
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  
  // 应用 mask 到 alpha 通道
  const rgbaData = Buffer.alloc(info.width * info.height * 4);
  
  for (let i = 0; i < info.width * info.height; i++) {
    rgbaData[i * 4] = rgbData[i * 4];         // R
    rgbaData[i * 4 + 1] = rgbData[i * 4 + 1]; // G
    rgbaData[i * 4 + 2] = rgbData[i * 4 + 2]; // B
    rgbaData[i * 4 + 3] = resizedMask[i];     // A (from mask)
  }
  
  return sharp(rgbaData, {
    raw: {
      width: info.width,
      height: info.height,
      channels: 4
    }
  });
}

/**
 * 裁剪透明边缘并调整大小
 */
async function trimAndResize(image) {
  console.log(`  ✂️ 裁剪透明边缘...`);
  
  // 先转为 buffer 以便处理
  const buffer = await image.png().toBuffer();
  
  try {
    const trimmed = await sharp(buffer)
      .trim({
        background: { r: 0, g: 0, b: 0, alpha: 0 },
        threshold: 10
      })
      .toBuffer({ resolveWithObject: true });
    
    let finalImage = sharp(trimmed.data);
    const { width, height } = trimmed.info;
    
    console.log(`  📐 裁剪后尺寸: ${width}x${height}`);
    
    if (width > MAX_SIZE || height > MAX_SIZE) {
      console.log(`  📏 缩放到最大 ${MAX_SIZE}px...`);
      finalImage = finalImage.resize(MAX_SIZE, MAX_SIZE, {
        fit: 'inside',
        withoutEnlargement: true
      });
    }
    
    return finalImage;
  } catch (error) {
    // 如果 trim 失败（例如图片全透明），返回原图
    console.log(`  ⚠️ 裁剪失败，使用原图: ${error.message}`);
    let finalImage = sharp(buffer);
    const metadata = await finalImage.metadata();
    
    if (metadata.width > MAX_SIZE || metadata.height > MAX_SIZE) {
      finalImage = finalImage.resize(MAX_SIZE, MAX_SIZE, {
        fit: 'inside',
        withoutEnlargement: true
      });
    }
    
    return finalImage;
  }
}

/**
 * 处理单个图片
 */
async function processImage(file, model) {
  const baseName = path.basename(file.name, path.extname(file.name));
  const outputPath = path.join(OUTPUT_DIR, `${baseName}.png`);
  
  console.log(`\n📷 处理: ${file.name}`);
  
  try {
    const maskInfo = await removeBackground(file.path, model);
    
    console.log(`  🎭 应用遮罩...`);
    const maskedImage = await applyMaskAndProcess(file.path, maskInfo);
    
    const finalImage = await trimAndResize(maskedImage);
    
    await finalImage.png().toFile(outputPath);
    
    console.log(`  ✅ 已保存: ${outputPath}`);
    return true;
  } catch (error) {
    console.error(`  ❌ 处理失败: ${error.message}`);
    console.error(error.stack);
    return false;
  }
}

/**
 * 格式化时间显示
 */
function formatHours(hours) {
  if (hours >= 24) {
    const days = Math.floor(hours / 24);
    const remainingHours = hours % 24;
    if (remainingHours === 0) {
      return `${days} 天`;
    }
    return `${days} 天 ${remainingHours} 小时`;
  }
  return `${hours} 小时`;
}

/**
 * 主函数
 */
async function main() {
  console.log('🚀 开始处理图片');
  console.log(`📅 处理 ${formatHours(HOURS)} 内的文件`);
  console.log(`📂 输入目录: ${INPUT_DIR}`);
  console.log(`📂 输出目录: ${OUTPUT_DIR}`);
  
  await fs.mkdir(OUTPUT_DIR, { recursive: true });
  
  const files = await getRecentFiles(INPUT_DIR, HOURS);
  
  if (files.length === 0) {
    console.log('\n📭 没有找到需要处理的文件');
    return;
  }
  
  console.log(`\n📋 找到 ${files.length} 个文件待处理:`);
  files.forEach(f => console.log(`   - ${f.name} (修改于: ${f.mtime.toLocaleString()})`));
  
  // 只加载模型（不需要 processor）
  console.log('\n🤖 加载 AI 模型...');
  const model = await AutoModel.from_pretrained('BritishWerewolf/IS-Net-Anime', {
    dtype: 'fp32',
  });
  console.log('✅ 模型加载完成');
  
  let successCount = 0;
  let failCount = 0;
  
  for (const file of files) {
    const success = await processImage(file, model);
    if (success) {
      successCount++;
    } else {
      failCount++;
    }
  }
  
  console.log('\n📊 处理完成:');
  console.log(`   ✅ 成功: ${successCount}`);
  console.log(`   ❌ 失败: ${failCount}`);
}

main().catch(console.error);
