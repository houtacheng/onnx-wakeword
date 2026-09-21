"""Simulate wakeword.js isDemo=true inference with real TTS audio."""
import numpy as np
import onnxruntime as ort, os, wave

# Absolute paths
MEL_ONNX = r'D:\code\voicute\onnx-wakeword\models\melspectrogram.onnx'
STD_ONNX = r'D:\code\voicute\onnx-wakeword\models\zh\doubaodoubao.onnx'
DEMO_ONNX = r'D:\code\voicute\onnx-wakeword\models\zh\doubaodoubao_test.onnx'
TTS_WAVS = [
    r'D:\code\voicute\kws-export\keywords\doubaodoubao_v9.3_zh\debug_samples\s0_orig.wav',
    r'D:\code\voicute\kws-export\keywords\doubaodoubao_v9.3_zh\debug_samples\s0_humanize_v1.wav',
    r'D:\code\voicute\kws-export\keywords\doubaodoubao_v9.3_zh\debug_samples\s0_far_v1.wav',
]

RAW_MELS, N_MELS, DURATION = 32, 34, 98

mel_sess = ort.InferenceSession(MEL_ONNX, providers=['CPUExecutionProvider'])
std_sess = ort.InferenceSession(STD_ONNX, providers=['CPUExecutionProvider'])
demo_sess = ort.InferenceSession(DEMO_ONNX, providers=['CPUExecutionProvider'])

for wav_path in TTS_WAVS:
    if not os.path.exists(wav_path):
        continue
    fname = os.path.basename(wav_path)
    print(f"\n{'='*60}")
    print(f"Testing: {fname}")
    print('='*60)

    # Load WAV
    wf = wave.open(wav_path, 'rb')
    sr = wf.getframerate()
    n_frames = wf.getnframes()
    audio_raw = wf.readframes(n_frames)
    wf.close()
    audio_np = np.frombuffer(audio_raw, dtype=np.int16).astype(np.float32) / 32768.0

    # ── Wakeword.js preprocess (line 156 of wakeword.js) ──
    # mel_in = new ort.Tensor('float32', audioData, [1, audioData.length])
    # melOut = await melSession.run({ input: melIn })
    # mel = melOut[Object.keys(melOut)[0]].data
    # const frames = Math.floor(mel.length / RAW_MELS)
    # const mel2d = new Float32Array(frames * RAW_MELS)
    # for (let i = 0; i < frames * RAW_MELS; i++) mel2d[i] = mel[i] / 10 + 2;
    mel_in = mel_sess.run(None, {'input': audio_np.reshape(1, -1)})[0]
    mel_flat = mel_in[0, 0]
    mel_preprocessed = mel_flat / 10.0 + 2.0  # Same as wakeword.js line 156
    frames = mel_preprocessed.shape[0]
    start = max(0, frames - DURATION)

    print(f"Mel shape: {mel_in.shape}, raw range=[{mel_flat.min():.2f}, {mel_flat.max():.2f}]")
    print(f"Preprocessed range=[{mel_preprocessed.min():.2f}, {mel_preprocessed.max():.2f}], frames={frames}")

    # ── Standard model path (isDemo=false) ──
    inp_std = np.zeros((1, DURATION, RAW_MELS), dtype=np.float32)
    for f_idx in range(DURATION):
        s = start + f_idx
        if 0 <= s < frames:
            inp_std[0, f_idx, :RAW_MELS] = mel_preprocessed[s]
    std_out = std_sess.run(None, {'input': inp_std})[0]
    prob_std = float(np.asarray(std_out).flatten()[0])

    # ── Demo model path (isDemo=true) — exact wakeword.js logic ──
    inp_demo = np.zeros((1, DURATION, N_MELS), dtype=np.float32)
    for f_idx in range(DURATION):
        s = start + f_idx
        if 0 <= s < frames:
            # Wakeword.js line 170: input.set(mel2d.subarray(s*32, (s+1)*32), f*34)
            inp_demo[0, f_idx, :RAW_MELS] = mel_preprocessed[s]
        # Last 2 cols stay zero (numpy array initialized to 0)
    demo_out = demo_sess.run(None, {'input': inp_demo})[0]
    prob_demo = float(np.asarray(demo_out).flatten()[0])

    # Verify padding is all zeros
    pad_col_32 = inp_demo[0, :, 32]
    pad_col_33 = inp_demo[0, :, 33]
    assert np.all(pad_col_32 == 0), "Padding col 32 not zero!"
    assert np.all(pad_col_33 == 0), "Padding col 33 not zero!"

    diff = abs(prob_std - prob_demo)
    print(f"\nStandard (98x32) -> sigmoid={prob_std:.6f}")
    print(f"Demo     (98x34) -> sigmoid={prob_demo:.6f}")
    status = 'OK' if diff < 0.05 else 'FAIL'
    print(f"Diff:       {diff:.6f} [{status}]")
