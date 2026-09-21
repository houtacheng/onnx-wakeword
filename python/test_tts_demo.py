"""Test demo model vs standard model with real TTS audio."""
import numpy as np
import onnxruntime as ort, json, os, sys, struct, wave

# Absolute paths - avoid __file__ resolution issues
BASE = r'D:\code\voicute\onnx-wakeword'
KWS_EXPORT = r'D:\code\voicute\kws-export'
sys.path.insert(0, KWS_EXPORT)

MEL_ONNX = os.path.join(BASE, 'models', 'melspectrogram.onnx')
STANDARD_ONNX = os.path.join(BASE, 'models', 'zh', 'doubaodoubao.onnx')
DEMO_ONNX = os.path.join(BASE, 'models', 'zh', 'doubaodoubao_test.onnx')
TTS_WAV = os.path.join(KWS_EXPORT, 'keywords', 'doubaodoubao_v9.3_zh', 'debug_samples', 's0_orig.wav')

print(f"Files:")
for p in [MEL_ONNX, STANDARD_ONNX, DEMO_ONNX, TTS_WAV]:
    print(f"  {p}: {'OK' if os.path.exists(p) else 'MISSING'}")

wav_f = wave.open(TTS_WAV, 'rb')
sr = wav_f.getframerate()
n_channels = wav_f.getnchannels()
n_frames = wav_f.getnframes()
audio_raw = wav_f.readframes(n_frames)
wav_f.close()
audio_np = np.frombuffer(audio_raw, dtype=np.int16).astype(np.float32) / 32768.0
print(f"\nAudio: {sr}Hz {n_channels}ch {n_frames/sr:.2f}s ({len(audio_np)} samples)")

if sr != 16000:
    import librosa
    audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=16000)
    print(f"Resampled to {16000}Hz: {len(audio_np)} samples")

# Mel spectrogram
mel_sess = ort.InferenceSession(MEL_ONNX, providers=['CPUExecutionProvider'])
mel_in = mel_sess.run(None, {'input': audio_np.reshape(1, -1)})[0]
mel_flat = mel_in[0, 0] / 10.0 + 2.0
frames = mel_flat.shape[0]
RAW_MELS, N_MELS, DURATION = 32, 34, 98
start = max(0, frames - DURATION)

print(f"\n=== Standard model (32 ch) ===")
inp_std = np.zeros((1, DURATION, RAW_MELS), dtype=np.float32)
for i in range(DURATION):
    s = start + i
    if 0 <= s < frames:
        inp_std[0, i, :RAW_MELS] = mel_flat[s]
std_sess = ort.InferenceSession(STANDARD_ONNX, providers=['CPUExecutionProvider'])
out_std = std_sess.run(None, {'input': inp_std})[0]
prob_std = float(np.asarray(out_std).flatten()[0])
print(f"Input shape: [1,{DURATION},{RAW_MELS}]")
print(f"Standard model prob={prob_std:.4f}")

print(f"\n=== Demo model (34 ch) ===")
inp_demo = np.zeros((1, DURATION, N_MELS), dtype=np.float32)
for i in range(DURATION):
    s = start + i
    if 0 <= s < frames:
        inp_demo[0, i, :RAW_MELS] = mel_flat[s]
demo_sess = ort.InferenceSession(DEMO_ONNX, providers=['CPUExecutionProvider'])
out_demo = demo_sess.run(None, {'input': inp_demo})[0]
prob_demo = float(np.asarray(out_demo).flatten()[0])
print(f"Input shape: [1,{DURATION},{N_MELS}]")
print(f"Demo model prob={prob_demo:.4f}")

print(f"\n=== Comparison ===")
diff = abs(prob_std - prob_demo)
print(f"Standard: {prob_std:.4f}, Demo: {prob_demo:.4f}, Diff: {diff:.4f}")
if diff < 0.05:
    print("OK - demo model matches standard within tolerance")
else:
    print(f"WARN - large difference! Possible bug.")
