# Affective Subtitler

A deep learning-based web application that analyzes audio/video files to perform **Speech-to-Text (ASR)** and **Speech Emotion Recognition (SER)**. It generates emotion-annotated subtitle files (.SRT) with precise timestamps and provides comprehensive emotion analysis visualizations.

## ✨ Key Features

### 1. Word-Level Speech Recognition
- **WhisperX Integration**: Advanced ASR with forced alignment for word-level accuracy
- **Automatic Language Detection**: Supports multiple languages with auto-detection
- **Precise Timestamping**: Sub-second accuracy for subtitle synchronization

### 2. Speech Emotion Recognition
- **Multi-Model Support**: Switch between emotion recognition models (SuperB, XLSR)
- **Emotion Post-Processing**: Smart tie-breaker logic favoring high-arousal emotions
- **Short Segment Handling**: Graceful handling of sub-0.5s audio segments

### 3. Comprehensive Visualizations
- **Emotion Timeline**: Interactive scatter plot showing emotion confidence over time
- **Pie Chart**: Overall emotion distribution statistics
- **Word Frequency Analysis**: 
  - Chinese text support with jieba tokenization
  - Top 10 most frequent words per emotion
  - Emotion-specific Word Clouds with adaptive color schemes
  - Comprehensive stop words filtering (Chinese + English)

### 4. Subtitle Generation
- **SRT Format**: Industry-standard subtitle files with emotion annotations
- **Emotion Tags**: Each subtitle line tagged with detected emotion
- **Download Ready**: One-click download with proper formatting

## 🛠️ Technical Stack

### Core Technologies
- **ASR Engine**: [WhisperX](https://github.com/m-bain/whisperX) - Whisper with forced alignment
- **SER Models**: Wav2Vec2-based transformers from Hugging Face
  - `superb/wav2vec2-base-superb-er`
  - `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition`
- **NLP**: 
  - `jieba` for Chinese word segmentation
  - Custom stop words filtering
- **Deep Learning Framework**: PyTorch 2.8.0 (CUDA 12.8 support)
- **Web Framework**: Streamlit
- **Visualization**: Plotly, Matplotlib, WordCloud

### Architecture Highlights
- **Hybrid Processing**: CPU-based ASR (stability) + GPU-based SER (speed)
- **Memory Management**: Explicit model cleanup and CUDA cache clearing
- **Error Handling**: Robust fallback mechanisms for various edge cases

## 🚀 Installation & Setup

### 1. Environment Setup
We recommend using a dedicated Conda environment to avoid dependency conflicts.

```bash
# Create and activate the environment
conda create -n affective_subtitler python=3.10 -y
conda activate affective_subtitler
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt --default-timeout=1000
```

**Key Dependencies:**
- `whisperx` - Word-level ASR
- `transformers` - SER models
- `streamlit` - Web UI
- `jieba` - Chinese tokenization
- `wordcloud` - Visual word frequency
- `plotly`, `matplotlib` - Interactive charts
- `torch`, `torchaudio` - Deep learning backend
- `librosa`, `soundfile` - Audio processing

### 3. CUDA Setup (Optional but Recommended)
The application automatically detects CUDA availability:
- **WhisperX (ASR)**: Runs on CPU (stability priority)
- **SER Models**: Runs on GPU if available (performance priority)

## 📖 Usage

### Running the Application
Use the provided helper script that configures environment variables:

```bash
# Make sure you are in the project directory
./run_app.sh
```

Alternatively, run directly:
```bash
streamlit run app.py
```

### Using the App
1. **Upload**: Open http://localhost:8501 and upload an audio/video file (.wav, .mp3, .mp4)
2. **Configure**: Select your preferred SER model from the sidebar
3. **Analyze**: Click "Start Analysis" and wait for processing
4. **Explore**: 
   - View emotion timeline scatter plot
   - Check pie chart for emotion distribution
   - Browse word frequency analysis per emotion
   - Download the generated .SRT subtitle file

## 🧠 How It Works

### Processing Pipeline
```
Audio/Video Input
    ↓
[WhisperX ASR] → Word-level transcription + timestamps
    ↓
[Audio Slicing] → Segment audio by detected speech boundaries
    ↓
[SER Analysis] → Analyze emotion for each segment
    ↓
[Post-Processing] → Apply tie-breaker logic, filter short segments
    ↓
[Word Frequency] → Chinese/English tokenization, stop words filtering
    ↓
[Visualization + SRT] → Generate charts, word clouds, and subtitle file
```

### Emotion Categories
The system detects the following emotions:
- **High Arousal**: Happy, Angry, Surprise/Fear
- **Low Arousal**: Sad, Neutral, Calm

### Post-Processing Logic
- **Tie-Breaker**: When top 2 emotions are within 5% confidence, prefer high-arousal emotions
- **Short Segment Filter**: Segments < 0.5s automatically labeled as "neutral"
- **Case Normalization**: All emotion labels normalized to lowercase for consistent processing

## 🎨 Visualization Features

### 1. Emotion Timeline
- X-axis: Time in seconds
- Y-axis: Confidence score (0-1)
- Color: Emotion type
- Size: Confidence magnitude
- Hover: Shows transcript text

### 2. Emotion Distribution Pie Chart
- Shows percentage of each emotion
- Interactive tooltips with exact counts
- Color-coded by emotion type

### 3. Content Emotion Word Frequency
- **Tabs**: One per detected emotion (sorted by frequency)
- **Top 10 List**: Ranked words with occurrence counts
- **Word Cloud**: 
  - Size represents frequency
  - Color scheme adapts to emotion type
  - Chinese character support via font auto-detection

## 🌐 Multilingual Support

### Chinese Language
- **Tokenization**: jieba for accurate word segmentation
- **Stop Words**: 50+ common Chinese stop words filtered
- **Font Rendering**: Auto-detects system Chinese fonts (Noto, PingFang, MS YaHei)

### English Language
- **Tokenization**: Built-in word splitting
- **Stop Words**: 70+ common English stop words filtered
- **Full Compatibility**: All features work seamlessly

## 📝 Models Used

### ASR Model
- **WhisperX** (base model)
- Word-level forced alignment
- Running on CPU with int8 quantization
- Automatic language detection

### SER Models
- **Default**: `superb/wav2vec2-base-superb-er`
- **Alternative**: `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition`
- Running on GPU with float16 precision (if available)

## ⚠️ Known Warnings

You may see the following warnings during startup, which can be **safely ignored**:

1. **PyAnnote Version Mismatch**
   ```
   Model was trained with pyannote.audio 0.0.1, yours is 3.4.0
   ```
   - Models still function correctly despite version differences

2. **PyTorch Version Mismatch**
   ```
   Model was trained with torch 1.10.0+cu102, yours is 2.8.0+cu128
   ```
   - Backward compatibility maintains functionality

3. **Streamlit Deprecations**
   ```
   Please replace `use_container_width` with `width`
   ```
   - UI displays correctly; will update in future versions

## 🔧 Configuration

### Environment Variables (via `run_app.sh`)
- `LD_LIBRARY_PATH`: Points to PyTorch's bundled CUDA libraries
- `PYTORCH_CUDA_ALLOC_CONF`: Optimizes CUDA memory allocation

### Model Selection
Use the sidebar dropdown to switch between SER models during runtime.

## 📊 Output Formats

### SRT Subtitle File
```
1
00:00:01,234 --> 00:00:03,456
[HAPPY] This is great news!

2
00:00:03,500 --> 00:00:05,678
[SAD] I'm disappointed about this.
```

## 🤝 Contributing

This project uses:
- Python 3.10
- Git for version control
- Conda for environment management

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- [WhisperX](https://github.com/m-bain/whisperX) for advanced ASR
- [Hugging Face](https://huggingface.co/) for SER models
- [Jieba](https://github.com/fxsjy/jieba) for Chinese NLP
- Streamlit community for the web framework
