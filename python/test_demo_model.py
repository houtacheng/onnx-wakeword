"""Quick test: load demo ONNX and verify 34-channel inference works."""
import numpy as np
import onnxruntime as ort, json, os, sys

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
KWS_EXPORT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'kws-export'))
sys.path.insert(0, KWS_EXPORT)

MEL_ONNX = os.path.join(BASE, 'models', 'melspectrogram.onnx')
TEST_ONNX = os.path.join(BASE, 'models', 'zh', 'doubaodoubao_test.onnx')
INFO_JSON = os.path.join(BASE, 'models', 'model_info_doubaodoubao_test.json')

mel_sess = ort.InferenceSession(MEL_ONNX, providers=['CPUExecutionProvider'])
demo_sess = ort.InferenceSession(TEST_ONNX, providers=['CPUExecutionProvider'])

with open(INFO_JSON) as f:
    info = json.load(f)

is_demo = bool(info.get('_warning'))
print(f"_warning detected: {is_demo}")
print(f"Demo model input shape: {demo_sess.get_inputs()[0].shape}")

# Generate dummy audio (matching training window)
AUDIO_WIN = 16192
audio = np.random.randn(AUDIO_WIN).astype(np.float32) * 0.1
mel_in = mel_sess.run(None, {'input': audio.reshape(1, -1)})[0]
mel_flat = mel_in[0, 0] / 10.0 + 2.0  # standard preprocessing
frames = mel_flat.shape[0]
RAW_MELS, N_MELS, DURATION = 32, 34, 98
start = max(0, frames - DURATION)

if is_demo:
    inp = np.zeros((1, DURATION, N_MELS), dtype=np.float32)
    for i in range(DURATION):
        s = start + i
        if 0 <= s < frames:
            inp[0, i, :RAW_MELS] = mel_flat[s]
else:
    inp = np.zeros((1, DURATION, RAW_MELS), dtype=np.float32)
    for i in range(DURATION):
        s = start + i
        if 0 <= s < frames:
            inp[0, i, :RAW_MELS] = mel_flat[s]

out = demo_sess.run(None, {'input': inp})[0]
prob = float(np.asarray(out).flatten()[0])
print(f"\nDemo model prob={prob:.4f} — OK")
