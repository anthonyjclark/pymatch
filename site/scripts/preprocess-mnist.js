import fs from "node:fs";
import http from "node:http";
import https from "node:https";
import path from "node:path";
import { fileURLToPath } from "node:url";
import zlib from "node:zlib";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PUBLIC_DATA_DIR = path.resolve(__dirname, "../public/data");
const MATCH_RESOURCES_DIR = path.resolve(__dirname, "../../match/resources");

const MIRRORS = [
  "https://storage.googleapis.com/cvdf-datasets/mnist/",
  "http://yann.lecun.com/exdb/mnist/",
];

const FILES = {
  train_x: "train-images-idx3-ubyte.gz",
  train_y: "train-labels-idx1-ubyte.gz",
  valid_x: "t10k-images-idx3-ubyte.gz",
  valid_y: "t10k-labels-idx1-ubyte.gz",
};

async function downloadFile(fileName) {
  const cachePath = path.join(PUBLIC_DATA_DIR, fileName);
  if (fs.existsSync(cachePath)) {
    return fs.readFileSync(cachePath);
  }

  for (const mirror of MIRRORS) {
    const url = mirror + fileName;
    try {
      const buffer = await new Promise((resolve, reject) => {
        const client = url.startsWith("https") ? https : http;
        client
          .get(url, (res) => {
            if (res.statusCode !== 200) {
              return reject(new Error(`Status ${res.statusCode}`));
            }
            const chunks = [];
            res.on("data", (chunk) => chunks.push(chunk));
            res.on("end", () => resolve(Buffer.concat(chunks)));
            res.on("error", reject);
          })
          .on("error", reject);
      });

      fs.writeFileSync(cachePath, buffer);
      return buffer;
    } catch (e) {
      continue;
    }
  }

  throw new Error(`Failed to download ${fileName}`);
}

async function preprocess() {
  fs.mkdirSync(PUBLIC_DATA_DIR, { recursive: true });
  fs.mkdirSync(MATCH_RESOURCES_DIR, { recursive: true });

  const binPathPublic = path.join(PUBLIC_DATA_DIR, "mnist.bin");
  const binPathResources = path.join(MATCH_RESOURCES_DIR, "mnist.bin");

  console.log("Fetching and preprocessing MNIST dataset into compressed float32 mnist.bin...");

  const txGz = await downloadFile(FILES.train_x);
  const tyGz = await downloadFile(FILES.train_y);
  const vxGz = await downloadFile(FILES.valid_x);
  const vyGz = await downloadFile(FILES.valid_y);

  const txBuf = zlib.gunzipSync(txGz).subarray(16);
  const tyBuf = zlib.gunzipSync(tyGz).subarray(8);
  const vxBuf = zlib.gunzipSync(vxGz).subarray(16);
  const vyBuf = zlib.gunzipSync(vyGz).subarray(8);

  const numTrain = tyBuf.length;
  const numValid = vyBuf.length;
  const numFeatures = 784;

  const header = Buffer.alloc(16);
  header.writeUInt32LE(0x4d4e4953, 0); // 'MNIS' magic number
  header.writeUInt32LE(numTrain, 4);
  header.writeUInt32LE(numValid, 8);
  header.writeUInt32LE(numFeatures, 12);

  const txFloats = new Float32Array(txBuf.length);
  for (let i = 0; i < txBuf.length; i++) {
    txFloats[i] = (txBuf[i] / 255.0 - 0.1307) / 0.3081;
  }

  const tyFloats = new Float32Array(tyBuf.length);
  for (let i = 0; i < tyBuf.length; i++) {
    tyFloats[i] = tyBuf[i];
  }

  const vxFloats = new Float32Array(vxBuf.length);
  for (let i = 0; i < vxBuf.length; i++) {
    vxFloats[i] = (vxBuf[i] / 255.0 - 0.1307) / 0.3081;
  }

  const vyFloats = new Float32Array(vyBuf.length);
  for (let i = 0; i < vyBuf.length; i++) {
    vyFloats[i] = vyBuf[i];
  }

  const rawPayload = Buffer.concat([
    header,
    Buffer.from(txFloats.buffer),
    Buffer.from(tyFloats.buffer),
    Buffer.from(vxFloats.buffer),
    Buffer.from(vyFloats.buffer),
  ]);

  const compressedPayload = zlib.gzipSync(rawPayload);

  fs.writeFileSync(binPathPublic, compressedPayload);
  fs.writeFileSync(binPathResources, compressedPayload);

  console.log(
    `Successfully generated compressed mnist.bin (${(compressedPayload.length / 1e6).toFixed(2)} MB)`
  );
}

preprocess().catch((err) => {
  console.warn("Preprocessing warning:", err.message);
});
