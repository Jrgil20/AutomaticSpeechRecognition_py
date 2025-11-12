import argparse
import time
from collections import deque

import soundcard as sc
import numpy as np


def transcribe_stream(model, recorder_ctx, samplerate, chunk_frames, out_path=None):
    """Capture audio chunks from recorder_ctx and send buffered audio to model.

    Prints incremental segments and appends text to out_path if provided.
    """
    # how many seconds of audio to keep in buffer for context
    buffer_seconds = 6.0
    max_chunks = max(1, int(buffer_seconds * samplerate / chunk_frames))
    buf = deque(maxlen=max_chunks)

    last_end = 0.0
    print("Recording... Press Ctrl+C to stop.")
    try:
        with recorder_ctx as recorder:
            while True:
                data = recorder.record(numframes=chunk_frames)
                # convert to mono (average channels) and ensure float32
                mono = np.mean(data, axis=1).astype(np.float32)
                buf.append(mono)

                # concatenate buffered audio for smoother context
                audio = np.concatenate(list(buf), axis=0)

                # Transcribe - faster-whisper accepts numpy arrays
                segments, _ = model.transcribe(audio, beam_size=5)

                # Print only newly produced segments (use start/end times)
                for segment in segments:
                    start = getattr(segment, "start", None)
                    end = getattr(segment, "end", None)
                    text = getattr(segment, "text", "").strip()
                    if text == "":
                        continue

                    # If start is available, only print segments that start after last_end
                    if start is None or start >= last_end - 1e-3:
                        print(text, end=" ", flush=True)
                        if out_path:
                            with open(out_path, "a", encoding="utf-8") as f:
                                f.write(text + " ")
                        # update last_end from segment end if available
                        if end is not None:
                            last_end = max(last_end, end)

                # small sleep to avoid busy loop (adjustable)
                time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nRecording stopped.")


def main():
    parser = argparse.ArgumentParser(description="Realtime capture + transcription (mic or loopback)")
    parser.add_argument("--source", choices=["mic", "loopback"], default="mic",
                        help="Audio source: 'mic' for microphone, 'loopback' to try capturing system output (what-you-hear)")
    parser.add_argument("--samplerate", type=int, default=16000)
    parser.add_argument("--chunk-seconds", type=float, default=0.5,
                        help="Chunk size in seconds for capture; smaller = lower latency")
    parser.add_argument("--device", type=str, default=None,
                        help="Optional device name to select a specific mic or loopback device")
    parser.add_argument("--out", type=str, default="transcript.txt",
                        help="Optional file to append transcription to")
    parser.add_argument("--device-model", type=str, default="base.en",
                        help="Faster-Whisper model name to load (e.g. base.en)")
    parser.add_argument("--device-cpu", action="store_true",
                        help="Force model to run on CPU (default) instead of attempting GPU")
    args = parser.parse_args()

    # Lazy import of model to avoid heavy imports when only CLI help or audio isn't used
    device = "cpu" if args.device_cpu else "cpu"

    # Test-file mode: transcribe a given audio file and exit (no soundcard needed)
    if getattr(args, "test_file", None):
        from faster_whisper import WhisperModel
        model = WhisperModel(args.device_model, device=device)
        print(f"Transcribing test file: {args.test_file}")
        segments, _ = model.transcribe(args.test_file, beam_size=5)
        for segment in segments:
            text = getattr(segment, "text", "").strip()
            if text:
                print(text, end=" ", flush=True)
                if args.out:
                    with open(args.out, "a", encoding="utf-8") as f:
                        f.write(text + " ")
        print("\nDone.")
        return

    # Initialize model for streaming
    from faster_whisper import WhisperModel
    model = WhisperModel(args.device_model, device=device)

    sr = args.samplerate
    chunk_frames = int(sr * args.chunk_seconds)

    # Choose recorder context based on source
    recorder_ctx = None
    if args.source == "mic":
        if args.device:
            mic = sc.get_microphone(args.device)
        else:
            mic = sc.default_microphone()
        recorder_ctx = mic.recorder(samplerate=sr)
    else:  # loopback
        # try to request a loopback-enabled microphone; fall back to default microphone
        try:
            if args.device:
                loop_mic = sc.get_microphone(args.device)
            else:
                # include_loopback may be supported by soundcard implementations
                loop_mic = sc.default_microphone(include_loopback=True)
            recorder_ctx = loop_mic.recorder(samplerate=sr)
        except Exception:
            print("Warning: loopback capture not available, falling back to default microphone.")
            mic = sc.default_microphone()
            recorder_ctx = mic.recorder(samplerate=sr)

    # start streaming transcription
    transcribe_stream(model, recorder_ctx, samplerate=sr, chunk_frames=chunk_frames, out_path=args.out)


if __name__ == "__main__":
    main()
