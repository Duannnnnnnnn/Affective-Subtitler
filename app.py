import streamlit as st
import os
import tempfile
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO
import utils

st.set_page_config(page_title="Affective Subtitler", layout="wide")

st.title("🎭 Affective Subtitler")
st.markdown("### Generate Subtitles with Emotion Recognition")

# 1. Input Section
st.sidebar.header("1. Upload File")
uploaded_file = st.sidebar.file_uploader("Upload Audio/Video", type=["wav", "mp3", "mp4"])

st.sidebar.header("2. Model Settings")
ser_model_option = st.sidebar.selectbox(
    "Select SER Model",
    (
        "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
        "superb/wav2vec2-base-superb-er"
    ),
    index=0,
    help="Choose the emotion recognition model. 'superb' might be more robust for general audio."
)

if uploaded_file is not None:
    # Save uploaded file to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1])
    tfile.write(uploaded_file.read())
    tfile.close()
    
    file_path = tfile.name
    
    st.sidebar.success(f"File uploaded: {uploaded_file.name}")
    
    # Play Audio
    st.audio(uploaded_file)
    
    if st.sidebar.button("Start Analysis"):
        with st.spinner("Analyzing audio... This may take a while (ASR + SER)..."):
            try:
                # Run the pipeline
                results = utils.process_audio_pipeline(file_path, ser_model_name=ser_model_option)
                
                # Store results in session state to persist
                st.session_state['results'] = results
                st.success("Analysis Complete!")
                
            except Exception as e:
                st.error(f"An error occurred: {e}")
                # Clean up temp file
                os.unlink(file_path)

    # 2. Visualization & Output Section
    if 'results' in st.session_state:
        results = st.session_state['results']
        
        # Prepare data for visualization
        df = pd.DataFrame(results)
        
        # Create three columns for better layout
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.subheader("2. Emotion Timeline")
            if not df.empty:
                # Create a scatter plot showing emotion confidence over time
                fig = px.scatter(
                    df, 
                    x="start", 
                    y="confidence", 
                    color="emotion", 
                    size="confidence", 
                    hover_data=["text"],
                    title="Emotion Timeline",
                    labels={"start": "Time (s)", "confidence": "Confidence Score"}
                )
                st.plotly_chart(fig, width='stretch')
            else:
                st.info("No speech segments detected.")
        
        with col2:
            st.subheader("Emotion Distribution")
            if not df.empty:
                # Pie Chart for emotion percentages
                emotion_counts = df['emotion'].value_counts().reset_index()
                emotion_counts.columns = ['emotion', 'count']
                fig_pie = px.pie(
                    emotion_counts, 
                    values='count', 
                    names='emotion',
                    title="Overall Emotion %"
                )
                st.plotly_chart(fig_pie, width='stretch')
            else:
                st.info("No data")
        
        with col3:
            st.subheader("High-Arousal Words")
            if not df.empty:
                # DEBUG: Show all detected emotions
                unique_emotions = df['emotion'].unique().tolist()
                st.caption(f"Detected: {', '.join(unique_emotions)}")
                
                # Normalize emotions to lowercase for filtering
                df['emotion_lower'] = df['emotion'].str.lower().str.strip()
                
                # Filter for high-arousal emotions (case-insensitive)
                # Include common variations: angry/anger, happy/joy, surprise/surprised
                high_arousal_keywords = ['angry', 'anger', 'happy', 'joy', 'surprise', 'surprised']
                high_arousal_df = df[df['emotion_lower'].isin(high_arousal_keywords)]
                
                if not high_arousal_df.empty:
                    # Combine all text from angry/happy segments
                    text_combined = " ".join(high_arousal_df['text'].tolist())
                    
                    if text_combined.strip():
                        # Generate Word Cloud with Chinese font support
                        import os
                        # Try to find a Chinese font
                        font_path = None
                        possible_fonts = [
                            '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
                            '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
                            '/System/Library/Fonts/PingFang.ttc',  # macOS
                            'C:\\Windows\\Fonts\\msyh.ttc',  # Windows Microsoft YaHei
                        ]
                        for font in possible_fonts:
                            if os.path.exists(font):
                                font_path = font
                                break
                        
                        wc_params = {
                            'width': 400,
                            'height': 400,
                            'background_color': 'white',
                            'colormap': 'Reds',
                            'regexp': r"[\w']+",  # Support unicode characters
                        }
                        if font_path:
                            wc_params['font_path'] = font_path
                        
                        wordcloud = WordCloud(**wc_params).generate(text_combined)
                        
                        # Display Word Cloud using matplotlib
                        fig_wc, ax = plt.subplots(figsize=(5, 5))
                        ax.imshow(wordcloud, interpolation='bilinear')
                        ax.axis('off')
                        ax.set_title("Angry/Happy Words", fontsize=12)
                        st.pyplot(fig_wc)
                        
                        st.caption(f"Words from {len(high_arousal_df)} segments")
                    else:
                        st.info("No text in high-arousal segments")
                else:
                    st.info("No high-arousal emotions detected")
            else:
                st.info("No data")

        # Transcript section (full width below visualizations)
        st.subheader("3. Transcript & Subtitles")
        
        col_download, col_transcript = st.columns([1, 3])
        
        with col_download:
            # Generate SRT
            srt_content = utils.generate_srt(results)
            
            # Download Button
            st.download_button(
                label="📥 Download .SRT File",
                data=srt_content,
                file_name=f"{os.path.splitext(uploaded_file.name)[0]}_emotion.srt",
                mime="text/plain"
            )
        
        with col_transcript:
            # Display Transcript
            st.markdown("#### Transcript Preview")
            for item in results:
                emotion_color = {
                    "happy": "green",
                    "sad": "blue",
                    "angry": "red",
                    "neutral": "grey",
                    "fear": "purple"
                }.get(item['emotion'], "black")
                
                st.markdown(
                    f"**[{utils.format_timestamp(item['start'])}]** "
                    f":{emotion_color}[[{item['emotion'].upper()}]] "
                    f"{item['text']}"
                )

else:
    st.info("Please upload an audio or video file to begin.")

# Cleanup logic could be added here or handled by OS temp cleaning
