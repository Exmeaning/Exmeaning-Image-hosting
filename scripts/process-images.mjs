// scripts/process-images.mjs
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
const DAYS = parseInt(process.env.DAYS || process.argv[2] || '7', 10);

// 支持的图片格式
const SUPPORTED_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.webp'];

/**
 * 获取指定天数内修改的文件
 */
async function getRecentFiles(dir, days) {
  const files = [];
  const cutoffTime = Date.now() - days * 24 * 60 * 60 * 1000;

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
  console.log(`  🔄 加载图片...`);
  
  // 读取原始图片
  const image = await RawImage.read(imagePath);
  const originalWidth = image.width;
  const originalHeight = image.height;
  
  console.log(`  📐 原始尺寸: ${originalWidth}x${originalHeight}`);
  
  // 处理图片
  const processed = await processor(image);
  
  // 运行模型获取 mask
  console.log(`  🤖 运行抠图模型...`);
  const output = await model({ input: processed.pixel_values });
  
  // 获取 mask 数据
  const maskData = output.output[0].data;
  const maskWidth = output.output[0].dims[2];
  const maskHeight = output.output[0].dims[1];
  
  return {
    maskData,
    maskWidth,
    maskHeight,
    originalWidth,
    originalHeight
  };
}

/**
 * 应用 mask 并处理图片
 */
async function applyMaskAndProcess(imagePath, maskInfo) {
  const { maskData, maskWidth, maskHeight, originalWidth, originalHeight } = maskInfo;
  
  // 将 mask 转换为图片
  const maskBuffer = Buffer.from(maskData);
  
  // 将 mask 调整为原始图片大小
  const resizedMask = await sharp(maskBuffer, {
    raw: {
      width: maskWidth,
      height: maskHeight,
      channels: 1
    }
  })
    .resize(originalWidth, originalHeight, { fit: 'fill' })
    .raw()
    .toBuffer();
  
  // 读取原始图片并转换为 RGBA
  const originalImage = sharp(imagePath);
  const { data: rgbData, info } = await originalImage
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
  
  // 创建带透明度的图片
  let processedImage = sharp(rgbaData, {
    raw: {
      width: info.width,
      height: info.height,
      channels: 4
    }
  });
  
  return processedImage;
}

/**
 * 裁剪透明边缘并调整大小
 */
async function trimAndResize(image) {
  console.log(`  ✂️ 裁剪透明边缘...`);
  
  // 裁剪透明边缘
  const trimmed = await image
    .trim({
      background: { r: 0, g: 0, b: 0, alpha: 0 },
      threshold: 10
    })
    .toBuffer({ resolveWithObject: true });
  
  let finalImage = sharp(trimmed.data);
  const { width, height } = trimmed.info;
  
  console.log(`  📐 裁剪后尺寸: ${width}x${height}`);
  
  // 如果尺寸超过 MAX_SIZE，进行缩放
  if (width > MAX_SIZE || height > MAX_SIZE) {
    console.log(`  📏 缩放到最大 ${MAX_SIZE}px...`);
    finalImage = finalImage.resize(MAX_SIZE, MAX_SIZE, {
      fit: 'inside',
      withoutEnlargement: true
    });
  }
  
  return finalImage;
}

/**
 * 处理单个图片
 */
async function processImage(file, model, processor) {
  const baseName = path.basename(file.name, path.extname(file.name));
  const outputPath = path.join(OUTPUT_DIR, `${baseName}.png`);
  
  console.log(`\n📷 处理: ${file.name}`);
  
  try {
    // 抠图
    const maskInfo = await removeBackground(file.path, model, processor);
    
    // 应用 mask
    console.log(`  🎭 应用遮罩...`);
    const maskedImage = await applyMaskAndProcess(file.path, maskInfo);
    
    // 裁剪和调整大小
    const finalImage = await trimAndResize(maskedImage);
    
    // 保存为 PNG
    await finalImage.png().toFile(outputPath);
    
    console.log(`  ✅ 已保存: ${outputPath}`);
    return true;
  } catch (error) {
    console.error(`  ❌ 处理失败: ${error.message}`);
    return false;
  }
}

/**
 * 主函数
 */
async function main() {
  console.log('🚀 开始处理图片');
  console.log(`📅 处理 ${DAYS} 天内的文件`);
  console.log(`📂 输入目录: ${INPUT_DIR}`);
  console.log(`📂 输出目录: ${OUTPUT_DIR}`);
  
  // 确保输出目录存在
  await fs.mkdir(OUTPUT_DIR, { recursive: true });
  
  // 获取需要处理的文件
  const files = await getRecentFiles(INPUT_DIR, DAYS);
  
  if (files.length === 0) {
    console.log('\n📭 没有找到需要处理的文件');
    return;
  }
  
  console.log(`\n📋 找到 ${files.length} 个文件待处理:`);
  files.forEach(f => console.log(`   - ${f.name} (修改于: ${f.mtime.toLocaleDateString()})`));
  
  // 加载模型（只加载一次）
  console.log('\n🤖 加载 AI 模型...');
  const processor = await AutoProcessor.from_pretrained('BritishWerewolf/IS-Net-Anime');
  const model = await AutoModel.from_pretrained('BritishWerewolf/IS-Net-Anime', {
    dtype: 'fp32',
  });
  console.log('✅ 模型加载完成');
  
  // 处理每个文件
  let successCount = 0;
  let failCount = 0;
  
  for (const file of files) {
    const success = await processImage(file, model, processor);
    if (success) {
      successCount++;
    } else {
      failCount++;
    }
  }
  
  // 输出统计
  console.log('\n📊 处理完成:');
  console.log(`   ✅ 成功: ${successCount}`);
  console.log(`   ❌ 失败: ${failCount}`);
}

main().catch(console.error);