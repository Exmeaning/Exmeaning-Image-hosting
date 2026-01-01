import { AutoModel, Tensor } from '@huggingface/transformers';
import sharp from 'sharp';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT_DIR = path.join(__dirname, '..');
const INPUT_DIR = path.join(ROOT_DIR, 'costume_origin');
const OUTPUT_DIR = path.join(ROOT_DIR, 'costume_icons');
const MAX_SIZE = 512;
const MODEL_SIZE = 1024;

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
 * 预处理：HWC -> CHW 并归一化到 [0, 1]
 */
async function prepareTensor(imagePath) {
  const { data } = await sharp(imagePath)
    .resize(MODEL_SIZE, MODEL_SIZE, { fit: 'fill' })
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });

  const numPixels = MODEL_SIZE * MODEL_SIZE;
  const float32Data = new Float32Array(3 * numPixels);

  for (let i = 0; i < numPixels; ++i) {
    float32Data[i] = (data[i * 3] / 255.0 - 0.5) / 0.5;               // R
    float32Data[i + numPixels] = (data[i * 3 + 1] / 255.0 - 0.5) / 0.5;     // G
    float32Data[i + 2 * numPixels] = (data[i * 3 + 2] / 255.0 - 0.5) / 0.5; // B
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
    console.log(`📷 处理: ${file.name}`);
    try {
      const metadata = await sharp(file.path).metadata();
      const imgTensor = await prepareTensor(file.path);

      // 执行推理，传入 img 键
      const outputs = await model({ img: imgTensor });

      // 根据你的报错日志，结构是 { mask: Tensor }
      const maskTensor = outputs.mask || outputs.output || outputs[0];

      if (!maskTensor || !maskTensor.data) {
        throw new Error('无法定位 mask 数据');
      }

      const maskData = maskTensor.data;

      // 创建灰度 Mask Buffer
      // IS-Net 输出的 dims 可能是 [1, 1, 1024, 1024] 或 [1, 1024, 1024]
      const maskBuffer = Buffer.alloc(MODEL_SIZE * MODEL_SIZE);
      for (let i = 0; i < maskData.length; i++) {
        // 模型输出通常是 0-1 的 Float32，转换为 0-255 Uint8
        maskBuffer[i] = Math.max(0, Math.min(255, Math.round(maskData[i] * 255)));
      }

      // 缩放 Mask 到原图尺寸
      const resizedMask = await sharp(maskBuffer, {
        raw: { width: MODEL_SIZE, height: MODEL_SIZE, channels: 1 }
      })
        .resize(metadata.width, metadata.height, { fit: 'fill' })
        .toBuffer();

      // 读取原图并合成
      const { data: rgbData } = await sharp(file.path)
        .ensureAlpha()
        .raw()
        .toBuffer({ resolveWithObject: true });

      const rgbaData = Buffer.alloc(metadata.width * metadata.height * 4);
      for (let i = 0; i < metadata.width * metadata.height; i++) {
        rgbaData[i * 4] = rgbData[i * 4];
        rgbaData[i * 4 + 1] = rgbData[i * 4 + 1];
        rgbaData[i * 4 + 2] = rgbData[i * 4 + 2];
        rgbaData[i * 4 + 3] = resizedMask[i];
      }

      const baseName = path.basename(file.name, path.extname(file.name));
      await sharp(rgbaData, {
        raw: { width: metadata.width, height: metadata.height, channels: 4 }
      })
        .trim({ background: { r: 0, g: 0, b: 0, alpha: 0 }, threshold: 10 })
        .resize(MAX_SIZE, MAX_SIZE, { fit: 'inside', withoutEnlargement: true })
        .png()
        .toFile(path.join(OUTPUT_DIR, `${baseName}.png`));

      console.log(`  ✅ 成功: ${baseName}.png`);
    } catch (error) {
      console.error(`  ❌ 失败: ${error.message}`);
    }
  }
}

main().catch(console.error);
