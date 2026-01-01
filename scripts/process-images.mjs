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

// Config based on preprocessor_config.json from BritishWerewolf/IS-Net-Anime
const NORM_MEAN = [0.485, 0.456, 0.406];
const NORM_STD = [1.0, 1.0, 1.0];

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
 * Preprocessing: Letterbox resize to 1024x1024 (contain) -> NCHW -> Normalize
 * Returns { tensor, layoutInfo } to help reconstruct the mask later
 */
async function prepareTensor(imagePath) {
  const meta = await sharp(imagePath).metadata();
  const scale = Math.min(MODEL_SIZE / meta.width, MODEL_SIZE / meta.height);
  const newW = Math.round(meta.width * scale);
  const newH = Math.round(meta.height * scale);

  // Sharp 'contain' automatically centers the image on the background
  // We reproduce the geometry logic to know where the content is for the mask
  // Note: sharp centers with rounding.

  const { data } = await sharp(imagePath)
    .resize(MODEL_SIZE, MODEL_SIZE, {
      fit: 'contain',
      background: { r: 0, g: 0, b: 0, alpha: 0 }
    })
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });

  const numPixels = MODEL_SIZE * MODEL_SIZE;
  const float32Data = new Float32Array(3 * numPixels);

  // Normalize: (x/255 - mean) / std
  for (let i = 0; i < numPixels; ++i) {
    const r = data[i * 3] / 255.0;
    const g = data[i * 3 + 1] / 255.0;
    const b = data[i * 3 + 2] / 255.0;

    float32Data[i] = (r - NORM_MEAN[0]) / NORM_STD[0];               // R
    float32Data[i + numPixels] = (g - NORM_MEAN[1]) / NORM_STD[1];   // G
    float32Data[i + 2 * numPixels] = (b - NORM_MEAN[2]) / NORM_STD[2]; // B
  }

  return {
    tensor: new Tensor('float32', float32Data, [1, 3, MODEL_SIZE, MODEL_SIZE]),
    layout: { width: newW, height: newH, origW: meta.width, origH: meta.height }
  };
}

async function main() {
  console.log('🚀 Starting image processing...');
  console.log(`Config: Mean=[${NORM_MEAN}], Std=[${NORM_STD}], Resize=Contain`);

  await fs.mkdir(OUTPUT_DIR, { recursive: true });

  const files = await getRecentFiles(INPUT_DIR, HOURS);
  if (files.length === 0) return console.log('📭 No new files found.');

  console.log(`🤖 Loading model (BritishWerewolf/IS-Net-Anime)...`);
  const model = await AutoModel.from_pretrained('BritishWerewolf/IS-Net-Anime', {
    dtype: 'fp32',
  });

  for (const file of files) {
    console.log(`📷 Processing: ${file.name}`);
    try {
      const { tensor, layout } = await prepareTensor(file.path);

      // Inference
      const outputs = await model({ img: tensor });
      const maskTensor = outputs.mask || outputs.output || outputs[0];

      if (!maskTensor || !maskTensor.data) {
        throw new Error('Mask output not found');
      }

      // Convert Mask Tensor (Float32) to Buffer (Uint8)
      // Mask is 1024x1024.
      const maskData = maskTensor.data;
      const maskBuffer = Buffer.alloc(MODEL_SIZE * MODEL_SIZE);
      for (let i = 0; i < maskData.length; i++) {
        maskBuffer[i] = Math.max(0, Math.min(255, Math.round(maskData[i] * 255)));
      }

      // 1. Create full 1024x1024 mask image
      let maskImg = sharp(maskBuffer, {
        raw: { width: MODEL_SIZE, height: MODEL_SIZE, channels: 1 }
      });

      // 2. Extract valid area (reverse the letterbox padding)
      // sharp 'contain' centers the image.
      const left = Math.floor((MODEL_SIZE - layout.width) / 2);
      const top = Math.floor((MODEL_SIZE - layout.height) / 2);

      const croppedMaskBuffer = await maskImg
        .extract({ left, top, width: layout.width, height: layout.height })
        .resize(layout.origW, layout.origH, { fit: 'fill' }) // Resize back to original dimensions
        .toBuffer();

      // 3. Composite with original image
      const { data: rgbData } = await sharp(file.path)
        .ensureAlpha()
        .raw()
        .toBuffer({ resolveWithObject: true });

      const rgbaData = Buffer.alloc(layout.origW * layout.origH * 4);

      // Merge original RGB with new Mask
      for (let i = 0; i < layout.origW * layout.origH; i++) {
        rgbaData[i * 4] = rgbData[i * 4];         // R
        rgbaData[i * 4 + 1] = rgbData[i * 4 + 1]; // G
        rgbaData[i * 4 + 2] = rgbData[i * 4 + 2]; // B
        rgbaData[i * 4 + 3] = croppedMaskBuffer[i]; // Alpha from mask
      }

      // 4. Final output: Trim transparent edges and resize to target icon size
      const baseName = path.basename(file.name, path.extname(file.name));
      await sharp(rgbaData, {
        raw: { width: layout.origW, height: layout.origH, channels: 4 }
      })
        .trim({ background: { r: 0, g: 0, b: 0, alpha: 0 }, threshold: 10 })
        .resize(MAX_SIZE, MAX_SIZE, { fit: 'inside', withoutEnlargement: true })
        .png()
        .toFile(path.join(OUTPUT_DIR, `${baseName}.png`));

      console.log(`  ✅ Saved: ${baseName}.png`);
    } catch (error) {
      console.error(`  ❌ Failed: ${error.message}`);
      console.error(error);
    }
  }
}

main().catch(console.error);
