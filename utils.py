import whisperx
import librosa
import torch
import numpy as np
from transformers import pipeline
import datetime
import gc

# Cache for multiple models
ser_pipelines = {}

def load_ser_model(model_name="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"):
    global ser_pipelines
    if model_name not in ser_pipelines:
        device = 0 if torch.cuda.is_available() else -1
        print(f"Loading SER model: {model_name} on device {device}...")
        # Using the audio-classification pipeline
        ser_pipelines[model_name] = pipeline("audio-classification", model=model_name, device=device)
    return ser_pipelines[model_name]

def transcribe_with_whisperx(file_path, model_name="base"):
    """
    Transcribes audio using WhisperX with word-level alignment.
    NOTE: Using CPU mode to avoid cuDNN compatibility issues with torch 2.8.0
    """
    # Force CPU mode to bypass cuDNN crash
    device = "cpu"
    compute_type = "int8"  # CPU-compatible compute type
    
    print(f"Loading WhisperX model: {model_name} on {device} ({compute_type})...")
    print("Note: Using CPU mode for WhisperX to avoid cuDNN compatibility issues")
    
    # 1. Transcribe with original whisper (batched)
    model = whisperx.load_model(model_name, device, compute_type=compute_type)
    audio = whisperx.load_audio(file_path)
    result = model.transcribe(audio, batch_size=16)
    
    # Explicitly delete model to free memory
    del model
    gc.collect()
    
    # 2. Align whisper output
    language_code = result["language"]
    print(f"Detected language: {language_code}")
    
    print("Loading alignment model...")
    model_a, metadata = whisperx.load_align_model(language_code=language_code, device=device)
    
    print("Aligning segments...")
    result_aligned = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    
    # Explicitly delete alignment model
    del model_a
    gc.collect()
        
    return result_aligned["segments"]

def analyze_emotion(audio_segment, sr, model_name="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"):
    """
    Analyzes emotion of a raw audio segment with post-processing.
    """
    # Short segment handling
    duration = len(audio_segment) / sr
    if duration < 0.5:
        return "neutral", 0.0

    pipe = load_ser_model(model_name)
    
    # Pipeline prediction
    # Ensure 16k
    if sr != 16000:
         # This might be redundant if we load at 16k, but good for safety
         pass 

    prediction = pipe({"array": audio_segment, "sampling_rate": 16000})
    
    # Post-processing logic
    # prediction is list of dicts sorted by score desc
    top1 = prediction[0]
    top2 = prediction[1]
    
    emotion = top1['label']
    confidence = top1['score']
    
    # Heuristic: if top 2 are close (< 5%), favor high arousal
    if (top1['score'] - top2['score']) < 0.05:
        # Normalize to lowercase for comparison
        high_arousal = ["happy", "angry", "fear", "surprise", "joy", "anger", "surprised"]
        top1_lower = top1['label'].lower().strip()
        top2_lower = top2['label'].lower().strip()
        
        if top2_lower in high_arousal and top1_lower not in high_arousal:
            emotion = top2['label']
            confidence = top2['score']
            print(f"Tie-breaker: Switched {top1['label']} -> {top2['label']}")
            
    return emotion, confidence

def process_audio_pipeline(file_path, ser_model_name="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"):
    """
    Full pipeline: WhisperX ASR -> Slicing -> SER -> Merge
    """
    # 1. ASR (WhisperX)
    print("Starting ASR (WhisperX)...")
    segments = transcribe_with_whisperx(file_path)
    
    # 2. Load Audio for Slicing
    print(f"Loading audio for SER (Model: {ser_model_name})...")
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
        if end_sample - start_sample < 800: # Skip extremely short (< 0.05s)
             continue

        audio_slice = y[start_sample:end_sample]
        
        # 3. SER
        try:
            emotion, confidence = analyze_emotion(audio_slice, sr, model_name=ser_model_name)
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
