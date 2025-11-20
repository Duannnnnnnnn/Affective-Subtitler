# Affective Subtitler

A deep learning-based Web application that analyzes user-uploaded audio/video files, performing both **Speech-to-Text (ASR)** and **Speech Emotion Recognition (SER)**. It generates subtitle files (.SRT) with precise timestamps and emotion tags, and provides an interactive emotion trend visualization report.

## Features

- **ASR (Speech-to-Text)**: Uses OpenAI Whisper for accurate transcription and timestamping.
- **SER (Speech Emotion Recognition)**: Uses Hugging Face Transformers (Wav2Vec2/HuBERT) to detect emotions in speech segments.
- **Visualization**: Interactive charts showing emotion trends over time.
- **Subtitle Generation**: Exports .SRT files with emotion annotations.

## Installation

1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: You might need to install `ffmpeg` on your system separately if not already available.*

## Usage

1.  Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```
2.  Upload an audio or video file (.wav, .mp3, .mp4).
3.  Click "Start Analysis".
4.  View the results and download the generated subtitle file.

## Models Used

- **ASR**: `openai/whisper-base`
- **SER**: `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition` (or similar)
