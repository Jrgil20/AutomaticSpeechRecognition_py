import soundcard as sc
import numpy as np
from faster_whisper import WhisperModel

def main():
    # Initialize the Whisper model
    model = WhisperModel("base.en", device="cpu")

    # Get the default microphone
    mic = sc.default_microphone()

    print("Recording... Press Ctrl+C to stop.")
    try:
        # Record from the microphone
        with mic.recorder(samplerate=16000) as recorder:
            while True:
                # Record a chunk of audio data
                data = recorder.record(numframes=4096)

                # The data is in stereo (2 channels), so we convert it to mono by taking the mean.
                # Whisper expects a single-channel (mono) audio stream.
                mono_data = np.mean(data, axis=1).astype(np.float32)

                # Transcribe the audio chunk
                segments, _ = model.transcribe(mono_data, beam_size=5)

                # Print the transcribed text
                for segment in segments:
                    print(segment.text, end="", flush=True)

    except KeyboardInterrupt:
        print("\nRecording stopped.")

if __name__ == "__main__":
    main()
