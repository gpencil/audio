import type { Rnnoise } from "@shiguredo/rnnoise-wasm";
import {
  compressLongSilences,
  drawWaveform,
  encodeWavMono16,
  floatTo16BitPCM,
  peakAbs,
  resampleTo48kMono,
  toMono,
  trimSilenceEdges,
} from "./audio-utils.js";

const RN_RATE = 48_000;
/** RNNoise 官方 demo 按 int16 幅值送入网络（short 强转为 float），与 Web Audio 的 ±1 浮点不一致，必须缩放。 */
const RNNOISE_PCM_SCALE = 32768;

const dropZone = document.getElementById("dropZone") as HTMLElement;
const fileInput = document.getElementById("fileInput") as HTMLInputElement;
const controls = document.getElementById("controls") as HTMLElement;
const mixSlider = document.getElementById("mix") as HTMLInputElement;
const mixVal = document.getElementById("mixVal") as HTMLElement;
const trimDbSlider = document.getElementById("trimDb") as HTMLInputElement;
const trimDbVal = document.getElementById("trimDbVal") as HTMLElement;
const trimEnable = document.getElementById("trimEnable") as HTMLInputElement;
const compressEnable = document.getElementById("compressEnable") as HTMLInputElement;
const compressOpts = document.getElementById("compressOpts") as HTMLElement;
const longMsSlider = document.getElementById("longMs") as HTMLInputElement;
const longMsVal = document.getElementById("longMsVal") as HTMLElement;
const shortMsSlider = document.getElementById("shortMs") as HTMLInputElement;
const shortMsVal = document.getElementById("shortMsVal") as HTMLElement;
const processBtn = document.getElementById("processBtn") as HTMLButtonElement;
const downloadBtn = document.getElementById("downloadBtn") as HTMLButtonElement;
const statusEl = document.getElementById("status") as HTMLElement;
const waveBefore = document.getElementById("waveBefore") as HTMLCanvasElement;
const waveAfter = document.getElementById("waveAfter") as HTMLCanvasElement;
const playerAfter = document.getElementById("playerAfter") as HTMLAudioElement;

let audioContext: AudioContext | null = null;
let decodedBuffer: AudioBuffer | null = null;
let processed48k: Float32Array | null = null;
let processedUrl: string | null = null;
let rnnoisePromise: Promise<Rnnoise> | null = null;

function getRnnoise(): Promise<Rnnoise> {
  if (!rnnoisePromise) {
    rnnoisePromise = import("@shiguredo/rnnoise-wasm").then((m) => m.Rnnoise.load());
  }
  return rnnoisePromise;
}

function setStatus(msg: string, isError = false) {
  statusEl.textContent = msg;
  statusEl.classList.toggle("error", isError);
}

function getAudioContext(): AudioContext {
  if (!audioContext) {
    audioContext = new AudioContext();
  }
  return audioContext;
}

async function decodeFile(file: File): Promise<AudioBuffer> {
  const ctx = getAudioContext();
  const ab = await file.arrayBuffer();
  return ctx.decodeAudioData(ab.slice(0));
}

function denoiseBlend(
  rnnoise: Rnnoise,
  samples48k: Float32Array,
  wet: number,
): Float32Array {
  const state = rnnoise.createDenoiseState();
  const frameSize = rnnoise.frameSize;
  const n = samples48k.length;
  const out = new Float32Array(n);
  const frame = new Float32Array(frameSize);
  const dry = 1 - wet;

  for (let i = 0; i < n; i += frameSize) {
    const remain = n - i;
    if (remain >= frameSize) {
      const drySlice = samples48k.subarray(i, i + frameSize);
      for (let j = 0; j < frameSize; j++) {
        frame[j] = drySlice[j] * RNNOISE_PCM_SCALE;
      }
      state.processFrame(frame);
      for (let j = 0; j < frameSize; j++) {
        const den = Math.max(-1, Math.min(1, frame[j] / RNNOISE_PCM_SCALE));
        out[i + j] = drySlice[j] * dry + den * wet;
      }
    } else if (remain > 0) {
      frame.fill(0);
      const dryPart = samples48k.subarray(i, i + remain);
      for (let j = 0; j < remain; j++) {
        frame[j] = dryPart[j] * RNNOISE_PCM_SCALE;
      }
      state.processFrame(frame);
      for (let j = 0; j < remain; j++) {
        const den = Math.max(-1, Math.min(1, frame[j] / RNNOISE_PCM_SCALE));
        out[i + j] = dryPart[j] * dry + den * wet;
      }
    }
  }

  state.destroy();
  return out;
}

async function runPipeline() {
  if (!decodedBuffer) {
    setStatus("请先上传音频。", true);
    return;
  }

  processBtn.disabled = true;
  downloadBtn.disabled = true;
  setStatus("正在加载降噪模型…");

  try {
    const rnnoise = await getRnnoise();
    setStatus("正在解码与重采样…");
    const mono = toMono(decodedBuffer);
    const working = await resampleTo48kMono(mono, decodedBuffer.sampleRate);

    const wet = Number(mixSlider.value) / 100;
    setStatus("正在 RNNoise 降噪（本地）…");
    let denoised = denoiseBlend(rnnoise, working, wet);

    const trimDb = Number(trimDbSlider.value);
    if (trimEnable.checked) {
      setStatus("正在裁切首尾静音…");
      denoised = trimSilenceEdges(denoised, RN_RATE, trimDb);
    }

    if (compressEnable.checked) {
      setStatus("正在压缩长静音段…");
      const longMs = Number(longMsSlider.value);
      const shortMs = Number(shortMsSlider.value);
      denoised = compressLongSilences(denoised, RN_RATE, trimDb, longMs, shortMs);
    }

    if (processedUrl) {
      URL.revokeObjectURL(processedUrl);
      processedUrl = null;
    }

    processed48k = denoised;
    const wav = encodeWavMono16(floatTo16BitPCM(denoised), RN_RATE);
    processedUrl = URL.createObjectURL(wav);
    playerAfter.src = processedUrl;

    drawWaveform(waveBefore, working, "rgba(139,147,167,0.85)");
    drawWaveform(waveAfter, denoised, "rgba(91,140,255,0.9)");

    const dur = (decodedBuffer.duration * 1000).toFixed(0);
    const peak = peakAbs(denoised);
    setStatus(
      `完成。原始时长约 ${dur} ms，输出 48 kHz 单声道 WAV；处理后峰值约 ${(20 * Math.log10(peak + 1e-12)).toFixed(1)} dBFS。`,
    );
    downloadBtn.disabled = false;
  } catch (e) {
    console.error(e);
    setStatus(e instanceof Error ? e.message : String(e), true);
  } finally {
    processBtn.disabled = false;
  }
}

function bindSliders() {
  mixSlider.addEventListener("input", () => {
    mixVal.textContent = `${mixSlider.value}%`;
  });
  trimDbSlider.addEventListener("input", () => {
    const v = Number(trimDbSlider.value);
    trimDbVal.textContent = `${v <= 0 ? "−" : ""}${Math.abs(v)} dB`;
  });
  longMsSlider.addEventListener("input", () => {
    longMsVal.textContent = longMsSlider.value;
  });
  shortMsSlider.addEventListener("input", () => {
    shortMsVal.textContent = shortMsSlider.value;
  });
  compressEnable.addEventListener("change", () => {
    compressOpts.hidden = !compressEnable.checked;
  });
}

dropZone.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", async () => {
  const f = fileInput.files?.[0];
  if (!f) return;
  await loadFile(f);
});

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("dragover");
});
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
dropZone.addEventListener("drop", async (e) => {
  e.preventDefault();
  dropZone.classList.remove("dragover");
  const f = e.dataTransfer?.files?.[0];
  if (f) await loadFile(f);
});

async function loadFile(file: File) {
  setStatus(`正在读取 ${file.name}…`);
  controls.hidden = true;
  try {
    decodedBuffer = await decodeFile(file);
    controls.hidden = false;
    processed48k = null;
    if (processedUrl) {
      URL.revokeObjectURL(processedUrl);
      processedUrl = null;
    }
    playerAfter.removeAttribute("src");
    downloadBtn.disabled = true;
    const mono = toMono(decodedBuffer);
    const preview = await resampleTo48kMono(mono, decodedBuffer.sampleRate);
    drawWaveform(waveBefore, preview, "rgba(139,147,167,0.85)");
    drawWaveform(waveAfter, new Float32Array(0), "rgba(91,140,255,0.9)");
    setStatus(
      `已加载：${file.name}，${decodedBuffer.sampleRate} Hz，${decodedBuffer.numberOfChannels} 声道，约 ${decodedBuffer.duration.toFixed(2)} 秒。点击「处理并试听」。`,
    );
  } catch (e) {
    console.error(e);
    setStatus(`无法解码该文件：${e instanceof Error ? e.message : String(e)}`, true);
  }
}

processBtn.addEventListener("click", () => void runPipeline());

downloadBtn.addEventListener("click", () => {
  if (!processedUrl) return;
  const a = document.createElement("a");
  a.href = processedUrl;
  a.download = "denoised-48k-mono.wav";
  a.click();
});

bindSliders();

window.addEventListener("resize", () => {
  if (decodedBuffer) {
    void (async () => {
      const mono = toMono(decodedBuffer!);
      const preview = await resampleTo48kMono(mono, decodedBuffer!.sampleRate);
      drawWaveform(waveBefore, preview, "rgba(139,147,167,0.85)");
      if (processed48k) {
        drawWaveform(waveAfter, processed48k, "rgba(91,140,255,0.9)");
      }
    })();
  }
});
