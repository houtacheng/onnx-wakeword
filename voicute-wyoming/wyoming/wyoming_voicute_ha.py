#!/usr/bin/env python3
"""Home Assistant compatible Wyoming server for Voicute ONNX models."""

import argparse
import asyncio
import logging
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
from wyoming.audio import AudioChunk, AudioChunkConverter, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Attribution, Describe, Info, WakeModel, WakeProgram
from wyoming.server import AsyncEventHandler, AsyncServer
from wyoming.wake import Detect, Detection, NotDetected

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from python.wakeword_engine import SAMPLE_RATE, WakeWordEngine

_LOGGER = logging.getLogger("voicute-wyoming")
_VERSION = "1.0.2"
_STRIDE_BYTES = (SAMPLE_RATE // 20) * 2


class Handler(AsyncEventHandler):
    def __init__(self, engine, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.engine = engine
        self.converter = AudioChunkConverter(rate=SAMPLE_RATE, width=2, channels=1)
        self.audio = bytearray()
        self.timestamp = 0
        self.detected = False
        self.names = None
        _LOGGER.info("Client connected")

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self._info().event())
        elif Detect.is_type(event.type):
            request = Detect.from_event(event)
            self.names = set(request.names) if request.names else None
            self._reset()
        elif AudioStart.is_type(event.type):
            self._reset()
        elif AudioChunk.is_type(event.type):
            chunk = self.converter.convert(AudioChunk.from_event(event))
            await self._process_audio(chunk.audio)
            self.timestamp += chunk.milliseconds
        elif AudioStop.is_type(event.type) and not self.detected:
            await self.write_event(NotDetected().event())
        return True

    def _reset(self):
        self.audio.clear()
        self.timestamp = 0
        self.detected = False

    async def _process_audio(self, pcm):
        self.audio.extend(pcm)
        needed = self.engine.audio_samples_needed * 2
        while len(self.audio) >= needed:
            window = bytes(self.audio[:needed])
            del self.audio[:_STRIDE_BYTES]
            samples = np.frombuffer(window, dtype=np.int16)
            result = self.engine.predict(samples.astype(np.float32))
            if result is None or (self.names and result["word"] not in self.names):
                continue
            rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))
            self.engine.l5_rms = rms
            self.engine.rms_hist[self.engine.l5_ri] = rms
            self.engine.rms_t_hist[self.engine.l5_ri] = time.time() * 1000
            self.engine.l5_ri = (self.engine.l5_ri + 1) % 128
            word = self.engine.detect(
                result["word"], result["prob"], result["cons_frames"]
            )
            if word:
                self.detected = True
                await self.write_event(
                    Detection(name=word, timestamp=int(self.timestamp)).event()
                )
                _LOGGER.info("Detected %s (%.1f%%)", word, result["prob"] * 100)

    def _info(self):
        owner = Attribution(
            name="Voicute", url="https://github.com/houtacheng/onnx-wakeword"
        )
        models = [WakeModel(
            name=item["name"], description=item["name"], phrase=item["name"],
            attribution=owner, installed=True, languages=["zh", "en"],
            version=_VERSION,
        ) for item in self.engine.models]
        return Info(wake=[WakeProgram(
            name="voicute", description="Voicute ONNX wake-word detection",
            attribution=owner, installed=True, version=_VERSION, models=models,
        )])


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", default="tcp://0.0.0.0:10400")
    parser.add_argument("--model-info", required=True)
    parser.add_argument("--mel", required=True)
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--cooldown", type=int, default=1500)
    parser.add_argument("--L1", type=int, default=1)
    parser.add_argument("--L3", type=int, default=1)
    parser.add_argument("--L5", type=int, default=0)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    engine = WakeWordEngine()
    engine.load(args.model_info, args.mel)
    engine.threshold = args.threshold
    engine.cooldown_ms = args.cooldown
    for name, value in (("L1", args.L1), ("L2", 0), ("L3", args.L3),
                        ("L4", 0), ("L5", args.L5)):
        setattr(engine, name, bool(value))
    _LOGGER.info("Keywords: %s", [item["name"] for item in engine.models])
    _LOGGER.info("Listening with standard Wyoming protocol on %s", args.uri)
    await AsyncServer.from_uri(args.uri).run(partial(Handler, engine))


if __name__ == "__main__":
    asyncio.run(main())
