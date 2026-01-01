import { AutoModel, AutoProcessor, RawImage } from '@huggingface/transformers';
import sharp from 'sharp';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT_DIR = path.join(__dirname, '..');
const INPUT_DIR = path.join(ROOT_DIR, 'costume_origin');
const OUTPUT_DIR = path.join(ROOT_DIR, 'costume_icons');
const MAX_SIZE = 512;

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
 * 使用 IS-Net-Anime 模型进行抠图
 */
async function removeBackground(imagePath, model, processor) {
  console.log(`  🔄 读取图片...`);
  const image = await RawImage.read(imagePath);
  
  // 获取原始尺寸用于后期还原
  const originalWidth = image.width;
  const originalHeight = image.height;
  console.log(`  📐 原始尺寸: ${originalWidth}x${originalHeight}`);
  
  console.log(`  🔧 预处理图片...`);
  const inputs = await processor(image);
  
  console.log(`  🤖 运行推理...`);
  // 核心修复：直接展开 inputs，它包含模型需要的正确 key (pixel_values)
  const outputs = await model(inputs);
  
  /**
   * IS-Net 处理：模型可能返回一个包含 'output' 或 'mask' 的对象
   * 或者直接返回 Tensor 数组。
   */
  let maskTensor = outputs.output || outputs.mask || outputs[0];
  
  // 确保我们拿到了数据。如果结果在数组里，取第一个。
  if (Array.isArray(maskTensor)) maskTensor = maskTensor[0];

  return {
    maskData: maskTensor.data, // Float32Array
    maskWidth: maskTensor.dims[maskTensor.dims.length - 1],
    maskHeight: maskTensor.dims[maskTensor.dims.length - 2],
    originalWidth,
    originalHeight
  };
}

/**
 * 应用 mask 并处理图片
 */
async function applyMaskAndProcess(imagePath, maskInfo) {
  const { maskData, maskWidth, maskHeight, originalWidth, originalHeight } = maskInfo;
  
  // 将 mask 数据 (0-1 float) 转换为 0-255 uint8
  const maskBuffer = Buffer.alloc(maskWidth * maskHeight);
  for (let i = 0; i < maskData.length; i++) {
    maskBuffer[i] = Math.max(0, Math.min(255, Math.round(maskData[i] * 255)));
  }
  
  // 将 mask 调整为原始图片大小
  const resizedMask = await sharp(maskBuffer, {
    raw: { width: maskWidth, height: maskHeight, channels: 1 }
  })
    .resize(originalWidth, originalHeight, { fit: 'fill' })
    .toBuffer();
  
  // 读取原始图片并合入 Alpha 通道
  const { data: rgbData } = await sharp(imagePath)
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  
  const rgbaData = Buffer.alloc(originalWidth * originalHeight * 4);
  for (let i = 0; i < originalWidth * originalHeight; i++) {
    rgbaData[i * 4] = rgbData[i * 4];
    rgbaData[i * 4 + 1] = rgbData[i * 4 + 1];
    rgbaData[i * 4 + 2] = rgbData[i * 4 + 2];
    rgbaData[i * 4 + 3] = resizedMask[i]; // 将模型输出作为 A 通道
  }
  
  return sharp(rgbaData, {
    raw: { width: originalWidth, height: originalHeight, channels: 4 }
  });
}

/**
 * 裁剪透明边缘并调整大小
 */
async function trimAndResize(image) {
  console.log(`  ✂️ 裁剪透明边缘...`);
  const buffer = await image.png().toBuffer();
  
  try {
    const trimmed = await sharp(buffer)
      .trim({ background: { r: 0, g: 0, b: 0, alpha: 0 }, threshold: 10 })
      .toBuffer({ resolveWithObject: true });
    
    let finalImage = sharp(trimmed.data);
    const { width, height } = trimmed.info;
    console.log(`  📐 裁剪后尺寸: ${width}x${height}`);
    
    if (width > MAX_SIZE || height > MAX_SIZE) {
      console.log(`  📏 缩放到 ${MAX_SIZE}px...`);
      finalImage = finalImage.resize(MAX_SIZE, MAX_SIZE, {
        fit: 'inside',
        withoutEnlargement: true
      });
    }
    return finalImage;
  } catch (error) {
    console.log(`  ⚠️ 裁剪失败: ${error.message}`);
    return sharp(buffer).resize(MAX_SIZE, MAX_SIZE, { fit: 'inside', withoutEnlargement: true });
  }
}

/**
 * 主函数
 */
async function main() {
  console.log('🚀 开始处理图片');
  await fs.mkdir(OUTPUT_DIR, { recursive: true });
  
  const files = await getRecentFiles(INPUT_DIR, HOURS);
  if (files.length === 0) {
    console.log('\n📭 没有找到需要处理的文件');
    return;
  }
  
  console.log(`\n📋 待处理: ${files.length} 个文件`);
  
  console.log('\n🤖 加载 AI 模型 (IS-Net-Anime)...');
  // 加载模型和处理器
  const model = await AutoModel.from_pretrained('BritishWerewolf/IS-Net-Anime', { dtype: 'fp32' });
  const processor = await AutoProcessor.from_pretrained('BritishWerewolf/IS-Net-Anime');
  console.log('✅ 加载完成');
  
  for (const file of files) {
    const baseName = path.basename(file.name, path.extname(file.name));
    const outputPath = path.join(OUTPUT_DIR, `${baseName}.png`);
    
    console.log(`\n📷 处理: ${file.name}`);
    try {
      const maskInfo = await removeBackground(file.path, model, processor);
      const maskedImage = await applyMaskAndProcess(file.path, maskInfo);
      const finalImage = await trimAndResize(maskedImage);
      
      await finalImage.png().toFile(outputPath);
      console.log(`  ✅ 已保存: ${outputPath}`);
    } catch (error) {
      console.error(`  ❌ 失败: ${error.message}`);
    }
  }
  console.log('\n📊 所有任务处理完毕');
}

main().catch(console.error);
