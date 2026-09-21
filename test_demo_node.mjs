/** Test wakeword.js demo model inference in Node.js (WASM-like environment). */
const ort = require('onnxruntime-node');
const fs = require('fs');

const BASE = __dirname;
const MEL_ONNX = `${BASE}/models/melspectrogram.onnx`;
const DEMO_ONNX = `${BASE}/models/zh/doubaodoubao_test.onnx`;
const INFO_JSON = JSON.parse(fs.readFileSync(`${BASE}/models/model_info_doubaodoubao_test.json`));

// Constants from wakeword.js
const RAW_MELS = 32;
const N_MELS = 34;
const DURATION = 98;

console.log('=== Config ===');
console.log(`_warning present: ${!!INFO_JSON._warning}`);
console.log(`Demo input shape (onnx):`, DEMO_ONNX && { batch: '?', time: 98, mels: 34 });

// Load sessions
const melSession = new ort.InferenceSession(fs.readFileSync(MEL_ONNX), { executionProviders: ['cpu'] });
const demoSession = new ort.InferenceSession(fs.readFileSync(DEMO_ONNX), { executionProviders: ['cpu'] });

// Generate dummy audio
const AUDIO_WIN = 16192;
const audioData = Array.from({ length: AUDIO_WIN }, () => (Math.random() - 0.5) * 0.1);
const audioTensor = new ort.Tensor('float32', new Float32Array(audioData), [1, AUDIO_WIN]);

// Mel spectrogram
const melOut = melSession.run({ input: audioTensor });
const melKey = Object.keys(melOut)[0];
const melRaw = melOut[melKey].data; // TypedArray of raw mel values

// Preprocess
const frames = Math.floor(melRaw.length / RAW_MELS);
const mel2d = new Float32Array(frames * RAW_MELS);
for (let i = 0; i < frames * RAW_MELS; i++) mel2d[i] = melRaw[i] / 10 + 2;

console.log(`\nMel: total=${melRaw.length} frames=${frames} mels/frame=${RAW_MELS}`);

// ── Simulate demo mode ──
const isDemo = true;
const start = Math.max(0, frames - DURATION);

let input;
if (isDemo) {
    input = new Float32Array(DURATION * N_MELS);  // all zeros initially
    for (let f = 0; f < DURATION; f++) {
        const s = start + f;
        if (s >= 0 && s < frames) {
            input.set(mel2d.subarray(s * RAW_MELS, (s + 1) * RAW_MELS), f * N_MELS);
        }
    }
} else {
    input = new Float32Array(DURATION * RAW_MELS);
    for (let f = 0; f < DURATION; f++) {
        const s = start + f;
        if (s >= 0 && s < frames) {
            input.set(mel2d.subarray(s * RAW_MELS, (s + 1) * RAW_MELS), f * RAW_MELS);
        }
    }
}

const inShape = isDemo ? [1, DURATION, N_MELS] : [1, DURATION, RAW_MELS];
console.log(`\nInput shape: ${inShape.join('x')} (non-zero elements: ${input.filter(v => v !== 0).length})`);

// Verify padding
let zeroCount = 0;
for (let t = 0; t < DURATION; t++) {
    if (input[t * N_MELS + 32] !== 0 || input[t * N_MELS + 33] !== 0) {
        console.log(`ERROR: non-zero padding at frame ${t}: col32=${input[t*N_MELS+32]} col33=${input[t*N_MELS+33]}`);
    } else {
        zeroCount++;
    }
}
console.log(`Zero padding verified: all ${zeroCount} frames have zeros at cols 32-33`);

// Inference
const inputTensor = new ort.Tensor('float32', input, inShape);
const out = demoSession.run({ input: inputTensor });
const outKey = Object.keys(out)[0];
const prob = out[outKey].data[0];

console.log(`\nDemo model prob=${prob.toFixed(6)} — ${Math.abs(prob) < 0.5 ? 'VALID' : 'SUSPICIOUS'}`);
console.log('\n✓ Demo model inference OK (Node.js/CPU)');
