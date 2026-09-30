#!/usr/bin/env python3
"""Build the Holdfast dev set in the FDB-v3 folder layout.

    python devset/build_devset.py                      # metadata + devset_benchmark.json only
    python devset/build_devset.py --tts                # + synthesize input.wav with OpenAI TTS
    python devset/build_devset.py --recordings DIR     # + import your own recordings DIR/<id>.(wav|m4a|mp3)

Each example becomes devset/data/<id>_<24 hex>/{input.wav, metadata.json}, which
run_tool_benchmark_all_released.py discovers like FDB-v3 data. Every input gets a
trailing ambient tail (default 25 s) like the benchmark's recordings, because the
harness only records the agent for the length of the input file.
"""

import argparse
import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
VOICES = ["alloy", "ash", "coral", "echo", "sage", "shimmer", "verse", "ballad"]


def scenario_record(s: dict) -> dict:
    return {
        "id": s["id"], "title": s["user"][:60], "domain": s["domain"], "difficulty": s["difficulty"],
        "disfluency_features": s.get("disfluency_features", []), "state_rollback_test": bool(s.get("rollback")),
        "expected_tool_calls": s["expected"], "num_expected_calls": len(s["expected"]),
        "dialogue": [{"user": s["user"], "ai": s["ai"]}], "latency_profile": "normal",
    }


def folder_name(sid: str) -> str:
    return f"{sid}_{hashlib.sha1(sid.encode()).hexdigest()[:24]}"


def to_pcm48k(src_bytes: bytes) -> np.ndarray:
    out = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", "pipe:0", "-f", "s16le", "-ac", "1", "-ar", "48000",
                          "pipe:1"], input=src_bytes, capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.int16)


def write_wav(path: Path, pcm: np.ndarray, tail_s: float, seed: int) -> None:
    rng = np.random.default_rng(seed)
    tail = (rng.standard_normal(int(tail_s * 48000)) * 40).astype(np.int16)  # faint room tone
    lead = (rng.standard_normal(int(0.5 * 48000)) * 40).astype(np.int16)
    audio = np.concatenate([lead, pcm, tail])
    import wave

    with wave.open(str(path), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(48000)
        w.writeframes(audio.tobytes())


def synthesize(text: str, voice: str) -> np.ndarray:
    from openai import OpenAI

    client = OpenAI()
    parts = text.split("[pause]")
    pcm = []
    for i, part in enumerate(parts):
        if part.strip():
            r = client.audio.speech.create(model="gpt-4o-mini-tts", voice=voice, input=part.strip(),
                                           instructions="Speak casually, like a real person talking to a voice "
                                                        "assistant, including the fillers and restarts as written.",
                                           response_format="wav")
            pcm.append(to_pcm48k(r.read()))
        if i < len(parts) - 1:
            pcm.append(np.zeros(int(1.8 * 48000), dtype=np.int16))
    return np.concatenate(pcm)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tts", action="store_true")
    ap.add_argument("--recordings", type=Path)
    ap.add_argument("--tail", type=float, default=25.0)
    args = ap.parse_args()

    scenarios = yaml.safe_load((HERE / "scenarios.yaml").read_text())["scenarios"]
    data = HERE / "data"
    records = []
    for i, s in enumerate(scenarios):
        rec = scenario_record(s)
        records.append(rec)
        d = data / folder_name(s["id"])
        d.mkdir(parents=True, exist_ok=True)
        (d / "metadata.json").write_text(json.dumps(rec, indent=2))
        if args.recordings:
            src = next((p for p in args.recordings.glob(f"{s['id']}.*")), None)
            if src:
                write_wav(d / "input.wav", to_pcm48k(src.read_bytes()), args.tail, i)
        elif args.tts and not (d / "input.wav").exists():
            print("synthesizing", s["id"], file=sys.stderr)
            write_wav(d / "input.wav", synthesize(s["user"], VOICES[i % len(VOICES)]), args.tail, i)
    (HERE / "devset_benchmark.json").write_text(
        json.dumps({"benchmark_name": "holdfast-devset", "version": "1", "scenarios": records}, indent=2))
    have = sum(1 for p in data.glob("*/input.wav"))
    print(f"{len(records)} scenarios, {have} with audio -> {data}")


if __name__ == "__main__":
    main()
