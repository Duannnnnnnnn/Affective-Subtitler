import utils
import numpy as np
import soundfile as sf
import os

def create_dummy_audio(filename="test_audio.wav", duration=5, sr=16000):
    print(f"Creating dummy audio: {filename}")
    t = np.linspace(0, duration, int(sr * duration))
    # Generate a sine wave (just to have some sound, though ASR might output gibberish or nothing)
    # To make ASR pick up something, we might need actual speech, but for pipeline testing, 
    # we just want to ensure it doesn't crash.
    # Let's try to generate silence/noise to see if it handles it, 
    # or if we can, we assume the user will test with real audio.
    # For this test, we just want to see if the functions run.
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    sf.write(filename, audio, sr)
    return filename

def test_pipeline():
    filename = create_dummy_audio()
    try:
        print("Testing pipeline...")
        # Note: Whisper might not detect any text in a sine wave, so segments might be empty.
        # This is expected. We just want to ensure models load and function runs.
        results = utils.process_audio_pipeline(filename)
        print("Pipeline finished successfully.")
        print(f"Results: {results}")
        
        # Test SRT generation even if empty
        srt = utils.generate_srt(results)
        print("SRT Generation finished.")
        print(srt)
        
    except Exception as e:
        print(f"Pipeline failed: {e}")
    finally:
        if os.path.exists(filename):
            os.remove(filename)

def test_ser_explicitly():
    print("Testing SER model loading explicitly...")
    try:
        # Create a dummy audio segment (1 second of noise)
        dummy_audio = np.random.uniform(-1, 1, 16000)
        
        # Test default model
        print("Testing default model...")
        emotion, confidence = utils.analyze_emotion(dummy_audio, 16000)
        print(f"Default SER Result: Emotion={emotion}, Confidence={confidence}")
        
        # Test superb model
        print("Testing superb model...")
        emotion, confidence = utils.analyze_emotion(dummy_audio, 16000, model_name="superb/wav2vec2-base-superb-er")
        print(f"Superb SER Result: Emotion={emotion}, Confidence={confidence}")
        
    except Exception as e:
        print(f"SER Test Failed: {e}")

if __name__ == "__main__":
    test_pipeline()
    test_ser_explicitly()
