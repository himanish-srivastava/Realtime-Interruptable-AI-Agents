# Dev set

20 **original** scenarios (not from FDB-v3) in the benchmark's format, covering all 12 tools,
all 5 disfluency types and 1 to 3 chained calls. Use them to tune without looking at test items.

```bash
.venv-agent/bin/python devset/build_devset.py --tts                 # quick: OpenAI TTS voices
.venv-agent/bin/python devset/build_devset.py --recordings my_recs/ # better: your own voices
SKIP_INSTALL=1 DATASET=devset ./reproduce.sh
```

For recordings, name each file after the scenario id (`dev_travel_02.m4a`, ...) and speak the
`user` line naturally. Real hesitations beat reading the fillers aloud exactly. A 25 s ambient
tail is appended because the harness records the agent only for the input file's duration.
TTS audio is cleaner than real speech, so treat TTS results as a smoke test, not a score estimate.
