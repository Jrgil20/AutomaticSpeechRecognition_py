# AutomaticSpeechRecognition_py

This repository contains a small real-time audio capture + transcription tool using `soundcard` and `faster-whisper`.

## Quick overview
- `transcriber.py` — main script. Can capture from microphone or attempt loopback (system output) and transcribe using faster-whisper.

## Installation
Create a virtual environment and install dependencies. Adjust `torch`/CUDA installs if you plan to use GPU.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# If you need GPU support, install a matching torch, e.g.:
# pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## Usage
Basic microphone capture (default):

```bash
python transcriber.py --source mic --samplerate 16000 --chunk-seconds 0.5 --out transcript.txt
```

Attempt to capture system output (loopback):

```bash
python transcriber.py --source loopback --samplerate 16000 --chunk-seconds 0.5 --out transcript.txt
```

Test mode (transcribe an existing audio file without audio hardware):

```bash
python transcriber.py --test-file example.wav --out transcript.txt
```

Help:

```bash
python transcriber.py --help
```

## Notes
- Loopback capture depends on OS and soundcard drivers. On Linux you may need to configure a virtual loopback device (PulseAudio/ALSA). On Windows, WASAPI loopback or "Stereo Mix" may be available.
- `faster-whisper` will download model files (e.g., `base.en`) when first used. This requires internet and disk space.
- For production or low-latency requirements, consider a streaming-capable backend or finer-grained chunk handling.

## Troubleshooting
- If you see errors importing `soundcard` or `faster-whisper`, ensure dependencies are installed and that binary wheels are compatible with your Python version and OS.
- If loopback isn't available, run with `--source mic` or use the `--test-file` mode to verify transcription works with a WAV sample.

## Next steps / Improvements
- Add an optional small web UI or socket to stream transcriptions to a client.
- Implement incremental streaming API to avoid re-transcribing overlapping context.
- Add unit tests that run `--test-file` against a small WAV and assert expected output (mock model or use a deterministic short audio).
