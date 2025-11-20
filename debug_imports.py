import numpy
print(f"Numpy version: {numpy.__version__}")

try:
    import pandas
    print(f"Pandas version: {pandas.__version__}")
except ImportError as e:
    print(f"Pandas failed: {e}")

try:
    import numba
    print(f"Numba version: {numba.__version__}")
except ImportError as e:
    print(f"Numba failed: {e}")

try:
    import librosa
    print(f"Librosa version: {librosa.__version__}")
except ImportError as e:
    print(f"Librosa failed: {e}")

try:
    import whisperx
    print("WhisperX imported successfully")
except ImportError as e:
    print(f"WhisperX failed: {e}")
