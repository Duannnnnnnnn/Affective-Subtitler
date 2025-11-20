# Affective Subtitler

A deep learning-based Web application that analyzes user-uploaded audio/video files, performing both **Speech-to-Text (ASR)** and **Speech Emotion Recognition (SER)**. It generates subtitle files (.SRT) with precise timestamps and emotion tags, and provides an interactive emotion trend visualization report.

## Features

- **ASR (Speech-to-Text)**: Uses OpenAI Whisper for accurate transcription and timestamping.
- **SER (Speech Emotion Recognition)**: Uses Hugging Face Transformers (Wav2Vec2/HuBERT) to detect emotions in speech segments.
- **Visualization**: Interactive charts showing emotion trends over time.
- **Subtitle Generation**: Exports .SRT files with emotion annotations.

## Installation & Setup

### 1. Environment Setup
We recommend using a dedicated Conda environment to manage dependencies and avoid conflicts.

```bash
# Create and activate the environment
conda create -n affective_subtitler python=3.10 -y
conda activate affective_subtitler
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt --default-timeout=1000
```
*Note: This includes WhisperX, Transformers, Streamlit, and other necessary libraries.*

## Usage

### Running the Application
We provide a helper script `run_app.sh` that automatically sets up the environment variables (especially for CUDA libraries) and launches the app.

```bash
# Make sure you are in the project directory
./run_app.sh
```

### Using the App
1.  Open the provided URL (usually http://localhost:8501).
2.  Upload an audio or video file (.wav, .mp3, .mp4).
3.  Click "Start Analysis".
4.  View the emotion trends and download the generated .SRT file.

## Models Used

- **ASR**: `WhisperX` (Word-level alignment, running on CPU to ensure stability)
- **SER**: `superb/wav2vec2-base-superb-er` (Running on GPU)

## Known Warnings
You may see the following warnings during startup, which can be safely ignored:
- `pyannote.audio 0.0.1 vs 3.4.0`
- `torch 1.10.0 vs 2.8.0`
These are due to version differences between the training environment of the pre-trained models and our current runtime environment.
