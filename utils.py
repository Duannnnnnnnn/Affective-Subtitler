import whisperx
import librosa
import torch
import numpy as np
from transformers import pipeline
from collections import Counter
import jieba
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
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int((td.microseconds / 1000))
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

def analyze_emotion_word_frequency(results, top_n=10):
    """
    Analyze word frequency per emotion with Chinese and English support.
    
    Args:
        results: List of dictionaries containing 'text', 'emotion', 'start', 'end'
        top_n: Number of top words to return per emotion (default: 10)
    
    Returns:
        Dictionary of {emotion: [(word, count), ...]} with top N words per emotion
    """
    # Comprehensive stop words for Chinese and English
    chinese_stop_words = set([
        '的', '了', '是', '我', '你', '他', '她', '它', '在', '有', '和', '就', '不',
        '人', '都', '一', '一個', '一个', '上', '下', '来', '去', '着', '过', '到', '说',
        '对', '为', '这', '那', '中', '个', '能', '好', '也', '会', '还', '要', '被',
        '从', '与', '及', '於', '于', '但', '很', '么', '吗', '呢', '吧', '啊', '哦',
        '嗯', '啦', '吗', '哪', '呀', '呢', '嘛', '喔', '哩', '哦', '所', '因', '于'
    ])
    
    english_stop_words = set([
        'the', 'is', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'that', 'this', 'it', 'be', 'are', 'was',
        'were', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'can', 'may', 'might', 'i', 'you', 'he', 'she', 'we', 'they',
        'my', 'your', 'his', 'her', 'its', 'our', 'their', 'me', 'him', 'them', 'us',
        'what', 'which', 'who', 'when', 'where', 'why', 'how', 'so', 'than', 'too',
        'just', 'if', 'about', 'into', 'through', 'during', 'before', 'after', 'above',
        'below', 'between', 'under', 'again', 'further', 'then', 'once', 'here', 'there',
        'all', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
        'only', 'own', 'same', 'than', 'very', 'well', 'um', 'uh', 'oh', 'yeah', 'yes',
        'no', 'okay', 'ok', 'like', 'know', 'think', 'mean', 'get', 'got', 'going'
    ])
    
    all_stop_words = chinese_stop_words | english_stop_words
    
    # Group text by emotion
    emotion_texts = {}
    for item in results:
        emotion = item['emotion'].lower().strip()
        text = item['text']
        
        if emotion not in emotion_texts:
            emotion_texts[emotion] = []
        emotion_texts[emotion].append(text)
    
    # Analyze word frequency for each emotion
    emotion_word_freq = {}
    
    for emotion, texts in emotion_texts.items():
        # Combine all text for this emotion
        combined_text = ' '.join(texts)
        
        # Tokenize using jieba for Chinese + English
        words = jieba.lcut(combined_text)
        
        # Filter: remove stop words, single characters (except important ones), and punctuation
        filtered_words = []
        for word in words:
            word_lower = word.lower().strip()
            # Skip if empty, stop word, or just punctuation
            if (word_lower and 
                word_lower not in all_stop_words and 
                not all(c in '，。！？；：""''（）【】《》、,,..!?;:\'"()[]<>/\\-_=+*&^%$#@~`' for c in word) and
                len(word) > 1):  # Keep words longer than 1 character
                filtered_words.append(word)
        
        # Count frequencies
        word_counts = Counter(filtered_words)
        
        # Get top N most common words
        emotion_word_freq[emotion] = word_counts.most_common(top_n)
    
    return emotion_word_freq

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
