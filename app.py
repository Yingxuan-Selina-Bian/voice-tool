import streamlit as st
import threading
import time
from audio_processor import (
    callback, recorder, process_audio_chunk, reorder_segments,
    group_segments, summarize_group, CHUNK_DURATION, SAMPLE_RATE,
    CHANNELS, CHUNK_SIZE
)
import sounddevice as sd
import whisper
import os
import queue
import numpy as np

# Initialize session state variables
if 'recording' not in st.session_state:
    st.session_state.recording = False
if 'buffer' not in st.session_state:
    st.session_state.buffer = []
if 'stream' not in st.session_state:
    st.session_state.stream = None
if 'record_thread' not in st.session_state:
    st.session_state.record_thread = None
if 'q' not in st.session_state:
    st.session_state.q = queue.Queue()
if 'model' not in st.session_state:
    st.session_state.model = whisper.load_model("small")

def start_recording():
    """Start the audio recording process"""
    if not st.session_state.recording:
        st.session_state.recording = True
        st.session_state.stream = sd.InputStream(
            callback=callback,
            channels=CHANNELS,
            samplerate=SAMPLE_RATE,
            blocksize=CHUNK_SIZE
        )
        st.session_state.stream.start()
        st.session_state.record_thread = threading.Thread(
            target=recorder,
            daemon=True
        )
        st.session_state.record_thread.start()

def stop_recording():
    """Stop the audio recording process"""
    if st.session_state.recording:
        st.session_state.recording = False
        if st.session_state.stream:
            st.session_state.stream.stop()
            st.session_state.stream.close()
        st.session_state.stream = None
        st.session_state.record_thread = None

def process_buffer():
    """Process the current buffer with semantic reordering and summarization"""
    if not st.session_state.buffer:
        st.warning("No audio segments to process!")
        return
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("OpenAI API key not found! Please set the OPENAI_API_KEY environment variable.")
        return
    
    # Get the time window from the slider
    time_window = st.session_state.time_window * 60  # Convert minutes to seconds
    
    # Reorder segments
    reordered = reorder_segments(st.session_state.buffer, api_key)
    
    # Group segments
    groups = group_segments(reordered, time_window)
    
    # Display results
    st.subheader("Semantically Reordered and Summarized Content")
    
    for i, group in enumerate(groups, 1):
        with st.expander(f"Group {i}"):
            # Show individual segments
            st.write("Segments:")
            for seg in group:
                st.write(f"[{seg['start']:.2f}s] {seg['text']}")
            
            # Show summary
            summary = summarize_group(group, api_key)
            st.write("Summary:")
            st.write(summary)

# Streamlit UI
st.title("Real-Time Audio Processing and Semantic Ordering")

# Recording controls
col1, col2 = st.columns(2)
with col1:
    if st.button("Start Recording", disabled=st.session_state.recording):
        start_recording()
with col2:
    if st.button("Stop Recording", disabled=not st.session_state.recording):
        stop_recording()

# Recording status
if st.session_state.recording:
    st.success("Recording in progress...")
else:
    st.info("Recording stopped")

# Time window slider for grouping
st.slider(
    "Group segments by time window (minutes)",
    min_value=1,
    max_value=30,
    value=5,
    key="time_window"
)

# Process button
if st.button("Process and Summarize"):
    process_buffer()

# Display current buffer
if st.session_state.buffer:
    st.subheader("Current Buffer")
    for i, seg in enumerate(st.session_state.buffer):
        st.write(f"{i+1}. [{seg['start']:.2f}s] {seg['text']}")

# Auto-refresh the page every 5 seconds when recording
if st.session_state.recording:
    time.sleep(5)
    st.experimental_rerun() 