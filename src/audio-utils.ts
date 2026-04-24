const RN_RATE = 48_000;

/** 混为单声道 Float32 */
export function toMono(audioBuffer: AudioBuffer): Float32Array {
  const n = audioBuffer.length;
  const ch = audioBuffer.numberOfChannels;
  if (ch === 1) {
    return audioBuffer.getChannelData(0).slice();
  }
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let c = 0; c < ch; c++) {
      s += audioBuffer.getChannelData(c)[i];
    }
    out[i] = s / ch;
  }
  return out;
}

/** 重采样到 48 kHz（RNNoise 要求） */
export async function resampleTo48kMono(mono: Float32Array, srcRate: number): Promise<Float32Array> {
  if (srcRate === RN_RATE) {
    return mono;
  }
  const lengthOut = Math.ceil((mono.length * RN_RATE) / srcRate);
  const offline = new OfflineAudioContext(1, lengthOut, RN_RATE);
  const buf = offline.createBuffer(1, mono.length, srcRate);
  buf.getChannelData(0).set(mono);
  const src = offline.createBufferSource();
  src.buffer = buf;
  src.connect(offline.destination);
  src.start(0);
  const rendered = await offline.startRendering();
  return rendered.getChannelData(0).slice();
}

/** 峰值（用于相对门限） */
export function peakAbs(samples: Float32Array): number {
  let p = 1e-12;
  for (let i = 0; i < samples.length; i++) {
    const a = Math.abs(samples[i]);
    if (a > p) p = a;
  }
  return p;
}

/** 分帧 RMS，hop 为样本数 */
function frameRmsAbove(
  samples: Float32Array,
  startSample: number,
  direction: 1 | -1,
  frame: number,
  hop: number,
  gate: number,
): boolean {
  let idx = startSample;
  const end = direction === 1 ? samples.length : -1;
  while (direction === 1 ? idx < end : idx > end) {
    const i0 = Math.max(0, Math.min(idx, samples.length - frame));
    let sum = 0;
    const lim = Math.min(frame, samples.length - i0);
    for (let j = 0; j < lim; j++) {
      const s = samples[i0 + j];
      sum += s * s;
    }
    const rms = Math.sqrt(sum / Math.max(1, lim));
    if (rms >= gate) return true;
    idx += direction * hop;
  }
  return false;
}

/**
 * 裁切首尾静音。thresholdDb 为相对峰值的 dB（如 −42 表示门限 = peak × 10^(−42/20)）。
 * paddingMs：在检测到的语音边界外各保留一点。
 */
export function trimSilenceEdges(
  samples: Float32Array,
  sampleRate: number,
  thresholdDb: number,
  paddingMs = 80,
): Float32Array {
  const peak = peakAbs(samples);
  const gate = peak * Math.pow(10, thresholdDb / 20);
  const frame = Math.max(64, Math.floor(sampleRate * 0.02));
  const hop = Math.floor(frame / 2);
  const pad = Math.floor((paddingMs / 1000) * sampleRate);

  let start = 0;
  for (let i = 0; i < samples.length; i += hop) {
    if (frameRmsAbove(samples, i, 1, frame, hop, gate)) {
      start = Math.max(0, i - pad);
      break;
    }
  }

  let end = samples.length;
  for (let i = samples.length - 1; i >= 0; i -= hop) {
    if (frameRmsAbove(samples, i, -1, frame, hop, gate)) {
      end = Math.min(samples.length, i + frame + pad);
      break;
    }
  }

  if (end <= start) {
    return samples.slice();
  }
  return samples.subarray(start, end).slice();
}

/**
 * 将「长于 longMs 的静音段」缩短为 shortMs（静音填零），减轻气口。
 */
export function compressLongSilences(
  samples: Float32Array,
  sampleRate: number,
  thresholdDb: number,
  longMs: number,
  shortMs: number,
): Float32Array {
  const peak = peakAbs(samples);
  const gate = peak * Math.pow(10, thresholdDb / 20);
  const frame = Math.max(64, Math.floor(sampleRate * 0.015));
  const hop = Math.floor(frame / 3);
  const longSamples = Math.floor((longMs / 1000) * sampleRate);
  const shortSamples = Math.floor((shortMs / 1000) * sampleRate);

  type Seg = { silent: boolean; len: number };
  const segs: Seg[] = [];
  let pos = 0;
  while (pos < samples.length) {
    const i0 = pos;
    let silent = true;
    const chunkEnd = Math.min(pos + hop * 8, samples.length);
    for (let i = pos; i < chunkEnd; i += hop) {
      const lim = Math.min(frame, samples.length - i);
      let sum = 0;
      for (let j = 0; j < lim; j++) {
        const s = samples[i + j];
        sum += s * s;
      }
      const rms = Math.sqrt(sum / Math.max(1, lim));
      if (rms >= gate) {
        silent = false;
        break;
      }
    }
    const next = Math.min(pos + hop * 8, samples.length);
    const len = next - i0;
    const last = segs[segs.length - 1];
    if (last && last.silent === silent) {
      last.len += len;
    } else {
      segs.push({ silent, len });
    }
    pos = next;
  }

  let outLen = 0;
  for (const s of segs) {
    if (s.silent && s.len > longSamples) {
      outLen += shortSamples;
    } else {
      outLen += s.len;
    }
  }

  const out = new Float32Array(outLen);
  let w = 0;
  let r = 0;
  for (const s of segs) {
    if (s.silent && s.len > longSamples) {
      w += shortSamples;
      r += s.len;
    } else {
      out.set(samples.subarray(r, r + s.len), w);
      w += s.len;
      r += s.len;
    }
  }
  return out;
}

export function floatTo16BitPCM(float32: Float32Array): Int16Array {
  const out = new Int16Array(float32.length);
  for (let i = 0; i < float32.length; i++) {
    const x = Math.max(-1, Math.min(1, float32[i]));
    out[i] = x < 0 ? Math.round(x * 0x8000) : Math.round(x * 0x7fff);
  }
  return out;
}

export function encodeWavMono16(pcm: Int16Array, sampleRate: number): Blob {
  const n = pcm.length;
  const buf = new ArrayBuffer(44 + n * 2);
  const view = new DataView(buf);
  const wStr = (o: number, s: string) => {
    for (let i = 0; i < s.length; i++) view.setUint8(o + i, s.charCodeAt(i));
  };
  wStr(0, "RIFF");
  view.setUint32(4, 36 + n * 2, true);
  wStr(8, "WAVE");
  wStr(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  wStr(36, "data");
  view.setUint32(40, n * 2, true);
  let off = 44;
  for (let i = 0; i < n; i++) {
    view.setInt16(off, pcm[i], true);
    off += 2;
  }
  return new Blob([buf], { type: "audio/wav" });
}

export function drawWaveform(canvas: HTMLCanvasElement, samples: Float32Array, color: string) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const dpr = window.devicePixelRatio || 1;
  const cssW = Math.max(1, canvas.clientWidth || canvas.offsetWidth || 400);
  const cssH = 72;
  canvas.width = Math.floor(cssW * dpr);
  canvas.height = Math.floor(cssH * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = "#0c0e14";
  ctx.fillRect(0, 0, cssW, cssH);
  if (samples.length === 0) return;
  const step = Math.max(1, Math.floor(samples.length / cssW));
  ctx.beginPath();
  ctx.strokeStyle = color;
  ctx.lineWidth = 1;
  const mid = cssH / 2;
  const amp = mid - 2;
  for (let x = 0; x < cssW; x++) {
    const i0 = Math.min(x * step, samples.length - 1);
    let min = 1;
    let max = -1;
    const i1 = Math.min(samples.length, i0 + step);
    for (let i = i0; i < i1; i++) {
      const s = samples[i];
      if (s < min) min = s;
      if (s > max) max = s;
    }
    ctx.moveTo(x + 0.5, mid - max * amp);
    ctx.lineTo(x + 0.5, mid - min * amp);
  }
  ctx.stroke();
}
