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
const MODEL_SIZE = 1024; // IS-Net 固定的输入尺寸

const HOURS = parseInt(process.env.HOURS || process.argv[2] || '168', 10);
const SUPPORTED_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.webp'];

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
        files.push({ path: filePath, name: entry.name, mtime: stats.mtime });
      }
    }
  } catch (e) { return []; }
  return files;
}

/**
 * 手动预处理：将图片转换为模型需要的 Float32 Tensor [1, 3, 1024, 1024]
 */
async function prepareTensor(imagePath) {
  const { data, info } = await sharp(imagePath)
    .resize(MODEL_SIZE, MODEL_SIZE, { fit: 'fill' })
    .ensureAlpha(1) // 确保有通道，方便处理
    .removeAlpha()  // 去掉透明度，只取 RGB
    .raw()
    .toBuffer({ resolveWithObject: true });

  const numPixels = MODEL_SIZE * MODEL_SIZE;
  const float32Data = new Float32Array(3 * numPixels);

  // 归一化并从 HWC (RGBRGB...) 转换为 CHW (RRR...GGG...BBB...)
  for (let i = 0; i < numPixels; ++i) {
    float32Data[i] = data[i * 3] / 255.0;           // R
    float32Data[i + numPixels] = data[i * 3 + 1] / 255.0; // G
    float32Data[i + 2 * numPixels] = data[i * 3 + 2] / 255.0; // B
  }

  return new Tensor('float32', float32Data, [1, 3, MODEL_SIZE, MODEL_SIZE]);
}

async function main() {
  console.log('🚀 开始处理图片');
  await fs.mkdir(OUTPUT_DIR, { recursive: true });
  
  const files = await getRecentFiles(INPUT_DIR, HOURS);
  if (files.length === 0) return console.log('📭 无新文件');

  console.log(`🤖 加载模型 (IS-Net-Anime)...`);
  const model = await AutoModel.from_pretrained('BritishWerewolf/IS-Net-Anime', {
    dtype: 'fp32',
  });

  for (const file of files) {
    console.log(`\n📷 处理: ${file.name}`);
    try {
      const metadata = await sharp(file.path).metadata();
      const imgTensor = await prepareTensor(file.path);

      // 核心修复：根据报错信息，输入键名必须为 "img"
      const outputs = await model({ img: imgTensor });
      
      // 获取输出数据
      const maskTensor = outputs.output || outputs[0];
      const maskData = maskTensor.data;

      // 构建 Mask Buffer
      const maskBuffer = Buffer.alloc(MODEL_SIZE * MODEL_SIZE);
      for (let i = 0; i < maskData.length; i++) {
        maskBuffer[i] = Math.round(Math.min(Math.max(maskData[i], 0), 1) * 255);
      }

      // 还原尺寸并合并
      const resizedMask = await sharp(maskBuffer, {
        raw: { width: MODEL_SIZE, height: MODEL_SIZE, channels: 1 }
      })
        .resize(metadata.width, metadata.height, { fit: 'fill' })
        .toBuffer();

      const { data: rgbData } = await sharp(file.path).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
      
      const rgbaData = Buffer.alloc(metadata.width * metadata.height * 4);
      for (let i = 0; i < metadata.width * metadata.height; i++) {
        rgbaData[i * 4] = rgbData[i * 4];
        rgbaData[i * 4 + 1] = rgbData[i * 4 + 1];
        rgbaData[i * 4 + 2] = rgbData[i * 4 + 2];
        rgbaData[i * 4 + 3] = resizedMask[i];
      }

      const baseName = path.basename(file.name, path.extname(file.name));
      await sharp(rgbaData, { raw: { width: metadata.width, height: metadata.height, channels: 4 } })
        .trim({ background: { r: 0, g: 0, b: 0, alpha: 0 }, threshold: 10 })
        .resize(MAX_SIZE, MAX_SIZE, { fit: 'inside', withoutEnlargement: true })
        .png()
        .toFile(path.join(OUTPUT_DIR, `${baseName}.png`));

      console.log(`  ✅ 成功`);
    } catch (error) {
      console.error(`  ❌ 失败: ${error.message}`);
    }
  }
}

main().catch(console.error);
