import whisper
import librosa
import torch
import numpy as np
from transformers import pipeline
import datetime

# Initialize models globally to avoid reloading
# In a real app, we might want to load these lazily or cache them
asr_model = None
ser_pipeline = None

def load_asr_model(model_name="base"):
    global asr_model
    if asr_model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Whisper model: {model_name} on {device}...")
        asr_model = whisper.load_model(model_name, device=device)
    return asr_model

def load_ser_model(model_name="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"):
    global ser_pipeline
    if ser_pipeline is None:
        device = 0 if torch.cuda.is_available() else -1
        print(f"Loading SER model: {model_name} on device {device}...")
        # Using the audio-classification pipeline
        ser_pipeline = pipeline("audio-classification", model=model_name, device=device)
    return ser_pipeline

def transcribe_audio(file_path):
    """
    Transcribes audio using Whisper and returns segments with timestamps.
    """
    model = load_asr_model()
    result = model.transcribe(file_path)
    return result["segments"]

def analyze_emotion(audio_segment, sr):
    """
    Analyzes emotion of a raw audio segment.
    """
    pipe = load_ser_model()
    # The pipeline expects a filename or numpy array.
    # If numpy array, it might need specific handling depending on the pipeline version,
    # but generally transformers pipelines handle numpy arrays if sampling rate is provided or assumed 16k.
    # However, for safety and compatibility, we might need to ensure it's in the right format.
    # The model 'ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition' is wav2vec2 based, usually 16kHz.
    
    # We need to ensure the audio is 16kHz for most wav2vec2 models
    if sr != 16000:
        # Resample if necessary (though we should probably load at 16k)
        audio_segment = librosa.resample(audio_segment, orig_sr=sr, target_sr=16000)
        
    # Pipeline expects a dict with "array" and "sampling_rate" or just the array if configured
    # Let's try passing the dict
    prediction = pipe({"array": audio_segment, "sampling_rate": 16000})
    
    # Prediction is usually a list of dicts [{'label': 'angry', 'score': 0.9}, ...]
    # We want the top one
    top_emotion = prediction[0]
    return top_emotion['label'], top_emotion['score']

def process_audio_pipeline(file_path):
    """
    Full pipeline: ASR -> Slicing -> SER -> Merge
    """
    # 1. ASR
    print("Starting ASR...")
    segments = transcribe_audio(file_path)
    
    # 2. Load Audio for Slicing
    # Load at 16k because SER model likely needs 16k
    print("Loading audio for SER...")
    y, sr = librosa.load(file_path, sr=16000)
    
    results = []
    
    print("Starting SER on segments...")
    for segment in segments:
        start_time = segment['start']
        end_time = segment['end']
        text = segment['text']
        
        # Calculate sample indices
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        
        # Slice audio
        # Add a small buffer or check bounds?
        # Whisper timestamps can sometimes be slightly off or segments very short.
        if end_sample - start_sample < 1600: # Skip very short segments (< 0.1s)
             results.append({
                "start": start_time,
                "end": end_time,
                "text": text,
                "emotion": "neutral", # Default/Fallback
                "confidence": 0.0
            })
             continue

        audio_slice = y[start_sample:end_sample]
        
        # 3. SER
        try:
            emotion, confidence = analyze_emotion(audio_slice, sr)
        except Exception as e:
            print(f"Error in SER for segment {start_time}-{end_time}: {e}")
            emotion = "unknown"
            confidence = 0.0
            
        results.append({
            "start": start_time,
            "end": end_time,
            "text": text,
            "emotion": emotion,
            "confidence": confidence
        })
        
    return results

def format_timestamp(seconds):
    """
    Formats seconds into SRT timestamp format: HH:MM:SS,mmm
    """
    td = datetime.timedelta(seconds=seconds)
    # Total seconds to hours, minutes, seconds, milliseconds
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    millis = int(td.microseconds / 1000)
    
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

def generate_srt(results):
    """
    Generates SRT content from the processed results.
    """
    srt_content = ""
    for i, item in enumerate(results):
        start_str = format_timestamp(item['start'])
        end_str = format_timestamp(item['end'])
        emotion_tag = f"[{item['emotion'].upper()}]"
        text = item['text'].strip()
        
        srt_content += f"{i+1}\n"
        srt_content += f"{start_str} --> {end_str}\n"
        srt_content += f"{emotion_tag} {text}\n\n"
        
    return srt_content
