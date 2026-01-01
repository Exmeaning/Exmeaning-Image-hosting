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

// 从环境变量或命令行参数获取小时数
const HOURS = parseInt(process.env.HOURS || process.argv[2] || '168', 10);

// 支持的图片格式
const SUPPORTED_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.webp'];

/**
 * 获取指定小时数内修改的文件
 */
async function getRecentFiles(dir, hours) {
  const files = [];
  const cutoffTime = Date.now() - hours * 60 * 60 * 1000; // 小时转毫秒

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
  
  const image = await RawImage.read(imagePath);
  const originalWidth = image.width;
  const originalHeight = image.height;
  
  console.log(`  📐 原始尺寸: ${originalWidth}x${originalHeight}`);
  
  const processed = await processor(image);
  
  console.log(`  🤖 运行抠图模型...`);
  const output = await model({ input: processed.pixel_values });
  
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
  
  const maskBuffer = Buffer.from(maskData);
  
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
  
  const originalImage = sharp(imagePath);
  const { data: rgbData, info } = await originalImage
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  
  const rgbaData = Buffer.alloc(info.width * info.height * 4);
  
  for (let i = 0; i < info.width * info.height; i++) {
    rgbaData[i * 4] = rgbData[i * 4];
    rgbaData[i * 4 + 1] = rgbData[i * 4 + 1];
    rgbaData[i * 4 + 2] = rgbData[i * 4 + 2];
    rgbaData[i * 4 + 3] = resizedMask[i];
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
  
  const trimmed = await image
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
}

/**
 * 处理单个图片
 */
async function processImage(file, model, processor) {
  const baseName = path.basename(file.name, path.extname(file.name));
  const outputPath = path.join(OUTPUT_DIR, `${baseName}.png`);
  
  console.log(`\n📷 处理: ${file.name}`);
  
  try {
    const maskInfo = await removeBackground(file.path, model, processor);
    
    console.log(`  🎭 应用遮罩...`);
    const maskedImage = await applyMaskAndProcess(file.path, maskInfo);
    
    const finalImage = await trimAndResize(maskedImage);
    
    await finalImage.png().toFile(outputPath);
    
    console.log(`  ✅ 已保存: ${outputPath}`);
    return true;
  } catch (error) {
    console.error(`  ❌ 处理失败: ${error.message}`);
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
  
  console.log('\n🤖 加载 AI 模型...');
  const processor = await AutoProcessor.from_pretrained('BritishWerewolf/IS-Net-Anime');
  const model = await AutoModel.from_pretrained('BritishWerewolf/IS-Net-Anime', {
    dtype: 'fp32',
  });
  console.log('✅ 模型加载完成');
  
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
  
  console.log('\n📊 处理完成:');
  console.log(`   ✅ 成功: ${successCount}`);
  console.log(`   ❌ 失败: ${failCount}`);
}

main().catch(console.error);