import streamlit as st
import os
import tempfile
import pandas as pd
import plotly.express as px
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
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("2. Emotion Trends")
            if not df.empty:
                # Create a scatter plot or line chart
                # We plot emotion confidence over time (using start time)
                # Color by emotion label
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
                st.plotly_chart(fig, use_container_width=True)
                
                # Also a simple line chart for dominant emotion confidence? 
                # Or maybe a bar chart of emotion counts
                st.markdown("#### Emotion Distribution")
                emotion_counts = df['emotion'].value_counts().reset_index()
                emotion_counts.columns = ['emotion', 'count']
                fig_bar = px.bar(emotion_counts, x='emotion', y='count', color='emotion')
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("No speech segments detected.")

        with col2:
            st.subheader("3. Transcript & Subtitles")
            
            # Generate SRT
            srt_content = utils.generate_srt(results)
            
            # Download Button
            st.download_button(
                label="Download .SRT File",
                data=srt_content,
                file_name=f"{os.path.splitext(uploaded_file.name)[0]}_emotion.srt",
                mime="text/plain"
            )
            
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
