import streamlit as st
import sounddevice as sd
import numpy as np
import time
import os
import wave
import threading
import queue
from dotenv import load_dotenv
import uuid
import tempfile
from deepgram import Deepgram

# Load environment variables
load_dotenv()

# Configuration
SAMPLE_RATE = 44100  # Higher sample rate for better quality
CHANNELS = 1
RECORDINGS_DIR = "recordings"

# Create recordings directory if it doesn't exist
os.makedirs(RECORDINGS_DIR, exist_ok=True)

# Initialize session state for recording status
if 'is_recording' not in st.session_state:
    st.session_state.is_recording = False
if 'recordings' not in st.session_state:
    st.session_state.recordings = []
if 'recording_start_time' not in st.session_state:
    st.session_state.recording_start_time = 0
if 'debug' not in st.session_state:
    st.session_state.debug = []
if 'audio_frames' not in st.session_state:
    st.session_state.audio_frames = []
if 'selected_device' not in st.session_state:
    st.session_state.selected_device = None

# Add debug message function
def add_debug(msg):
    """Add debug message to session state for tracking"""
    st.session_state.debug.append(f"{time.strftime('%H:%M:%S')}: {msg}")

# Function to get audio devices
def get_audio_devices():
    """Get list of available audio input devices"""
    devices = sd.query_devices()
    input_devices = []
    
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            input_devices.append({
                'index': i,
                'name': device['name'],
                'channels': device['max_input_channels'],
                'default': device.get('default_input', False)
            })
    
    add_debug(f"Found {len(input_devices)} input devices")
    return input_devices

# Simple fixed-duration recording approach
def record_audio_simple(duration, device_index=None):
    """Record audio for a fixed duration"""
    add_debug(f"Starting simple recording for {duration} seconds" + 
             (f" on device {device_index}" if device_index is not None else ""))
    
    try:
        # Use the blocking recording approach
        recording = sd.rec(
            int(duration * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            device=device_index,
            dtype='int16'
        )
        # Wait for the recording to complete
        sd.wait()
        
        add_debug(f"Finished recording, shape: {recording.shape}")
        return recording
    except Exception as e:
        add_debug(f"Error in simple recording: {str(e)}")
        raise e

def save_audio(audio_data, filename):
    """Save audio data to a WAV file"""
    filepath = os.path.join(RECORDINGS_DIR, filename)
    with wave.open(filepath, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 16-bit audio (2 bytes)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_data.tobytes())
    
    # Get absolute path for the file
    abs_path = os.path.abspath(filepath)
    add_debug(f"Saved audio to {abs_path} ({os.path.getsize(filepath)} bytes)")
    
    return filepath

def transcribe_audio(filepath, api_key):
    """Transcribe audio file using Deepgram"""
    add_debug(f"Transcribing audio file: {filepath}")
    
    # Initialize Deepgram client
    dg = Deepgram(api_key)
    
    # Open the audio file
    with open(filepath, 'rb') as audio:
        # Set source parameters
        source = {'buffer': audio, 'mimetype': 'audio/wav'}
        
        # Send request to Deepgram
        response = dg.transcription.prerecorded(
            source,
            {
                'smart_format': True,
                'model': 'nova-2',
                'language': 'en',
                'punctuate': True,
                'diarize': False,
                'utterances': True,
                'word_timestamps': True
            }
        )
    
    # Extract words with timestamps
    words = []
    if 'results' in response and 'channels' in response['results']:
        for channel in response['results']['channels']:
            if 'alternatives' in channel and channel['alternatives']:
                transcript_text = channel['alternatives'][0]['transcript']
                if 'words' in channel['alternatives'][0]:
                    words = [{
                        'text': word['word'],
                        'start': word['start'],
                        'end': word['end']
                    } for word in channel['alternatives'][0]['words']]
    
    add_debug(f"Transcription completed: {transcript_text[:30]}..." if transcript_text else "No transcription")
    
    return {
        'text': transcript_text if transcript_text else "No transcription available",
        'words': words
    }

# Streamlit UI
st.title("Voice Recorder")

# Audio device selection
devices = get_audio_devices()
device_options = ["Default"] + [f"{d['name']} (Index: {d['index']})" for d in devices]

with st.expander("Audio Settings", expanded=False):
    selected_device_name = st.selectbox(
        "Select Microphone",
        options=device_options,
        index=0
    )
    
    # Extract device index from selection
    if selected_device_name != "Default":
        # parse the numeric index out of the string
        idx = int(selected_device_name.split("Index: ")[1].rstrip(")"))
        st.session_state.selected_device = idx

        # Set sounddevice defaults
        sd.default.device = idx
        sd.default.samplerate = SAMPLE_RATE
        sd.default.channels = CHANNELS

        st.write(f"Now using input device #{idx}: \"{selected_device_name}\"")
    else:
        st.session_state.selected_device = None
        # reset to system default
        sd.default.device = None
        st.write("Now using system default input device")
    
    # Test audio button
    if st.button("Test Microphone"):
        with st.spinner("Testing microphone for 3 seconds..."):
            try:
                test_recording = record_audio_simple(3, st.session_state.selected_device)
                
                # Save to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as f:
                    temp_file = f.name
                
                with wave.open(temp_file, 'wb') as wf:
                    wf.setnchannels(CHANNELS)
                    wf.setsampwidth(2)
                    wf.setframerate(SAMPLE_RATE)
                    wf.writeframes(test_recording.tobytes())
                
                st.audio(temp_file)
                st.success("Microphone test successful!")
                add_debug(f"Microphone test successful, audio shape: {test_recording.shape}")
            except Exception as e:
                st.error(f"Microphone test failed: {str(e)}")
                add_debug(f"Microphone test failed: {str(e)}")

# API key input
with st.expander("API Settings", expanded=False):
    api_key = st.text_input(
        "Enter your Deepgram API key", 
        value=os.getenv("DEEPGRAM_API_KEY", ""), 
        type="password"
    )

    if api_key:
        os.environ["DEEPGRAM_API_KEY"] = api_key
        add_debug("API key set")

# Recording controls
col1, col2 = st.columns(2)

# Recording duration slider
recording_duration = st.slider("Recording Duration (seconds)", 3, 60, 10)

with col1:
    if st.button("Start Recording", disabled=st.session_state.is_recording, key="start_btn", use_container_width=True):
        if not api_key:
            st.error("Please enter your Deepgram API key first")
        else:
            st.session_state.is_recording = True
            st.session_state.recording_start_time = time.time()
            add_debug("Started recording process")
            st.rerun()

with col2:
    if st.button("Stop Recording", disabled=not st.session_state.is_recording, key="stop_btn", use_container_width=True):
        if st.session_state.is_recording:
            st.session_state.is_recording = False
            add_debug("Stopped recording process")
            
            # Get the elapsed time
            elapsed = time.time() - st.session_state.recording_start_time
            actual_duration = min(elapsed, recording_duration)
            
            try:
                # Record audio using simple approach
                audio_data = record_audio_simple(
                    actual_duration, 
                    st.session_state.selected_device
                )
                
                if len(audio_data) > 0:
                    add_debug(f"Recorded audio with shape: {audio_data.shape}")
                    
                    # Create a unique filename
                    filename = f"recording_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.wav"
                    
                    # Save audio file
                    filepath = save_audio(audio_data, filename)
                    
                    # Check if file exists and has content
                    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
                        # Transcribe audio
                        with st.spinner("Transcribing audio..."):
                            transcription = transcribe_audio(filepath, api_key)
                        
                        # Create recording data
                        recording = {
                            'id': uuid.uuid4().hex,
                            'filename': filename,
                            'filepath': filepath,
                            'duration': actual_duration,
                            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                            'transcription': transcription
                        }
                        
                        # Add to recordings list
                        st.session_state.recordings.append(recording)
                        add_debug(f"Added new recording: {filename}")
                        
                        st.success(f"Recording saved and transcribed! Duration: {actual_duration:.2f} seconds")
                    else:
                        st.error(f"Failed to save audio file or file is empty")
                        add_debug(f"File error: exists={os.path.exists(filepath)}, size={os.path.getsize(filepath) if os.path.exists(filepath) else 0}")
                else:
                    st.warning("No audio was recorded. Please check your microphone.")
                    add_debug("No audio data captured")
            except Exception as e:
                st.error(f"Error recording audio: {str(e)}")
                add_debug(f"Recording error: {str(e)}")
            
            st.rerun()

# Recording status
if st.session_state.is_recording:
    elapsed_time = time.time() - st.session_state.recording_start_time
    remaining = max(0, recording_duration - elapsed_time)
    
    progress = min(1.0, elapsed_time / recording_duration)
    st.progress(progress)
    
    st.write(f"Recording... {elapsed_time:.1f}s / {recording_duration}s (Remaining: {remaining:.1f}s)")
    
    # Auto-refresh while recording to update timer
    time.sleep(0.1)
    st.rerun()

# Display recordings count
st.write(f"Number of recordings: {len(st.session_state.recordings)}")

# Display all recordings
for i, recording in enumerate(reversed(st.session_state.recordings)):
    st.markdown("---")
    st.subheader(f"Recording {len(st.session_state.recordings) - i}")
    
    # Display the audio file
    try:
        st.audio(recording['filepath'])
        st.write(f"📅 {recording['timestamp']} | ⏱️ {recording['duration']:.2f}s")
        st.write(f"📝 {recording['transcription']['text']}")
    except Exception as e:
        st.error(f"Error playing audio: {str(e)}")
        st.write(f"File path: {recording['filepath']}")
        st.write(f"File exists: {os.path.exists(recording['filepath'])}")
        
    # Add a download button for each recording
    try:
        with open(recording['filepath'], "rb") as file:
            st.download_button(
                label="Download Audio",
                data=file,
                file_name=recording['filename'],
                mime="audio/wav",
                key=f"download_{i}"
            )
    except Exception as e:
        st.error(f"Error creating download button: {str(e)}")

# Debug information in expander
with st.expander("Debug Information", expanded=False):
    st.write("Debug Log:")
    for msg in st.session_state.debug:
        st.write(msg)
    
    st.write("Recordings Directory:")
    if os.path.exists(RECORDINGS_DIR):
        files = os.listdir(RECORDINGS_DIR)
        for file in files:
            file_path = os.path.join(RECORDINGS_DIR, file)
            st.write(f"{file}: {os.path.getsize(file_path)} bytes")
    else:
        st.write("Recordings directory doesn't exist!")
    
    if st.button("Clear All Recordings"):
        st.session_state.recordings = []
        add_debug("Cleared all recordings")
        st.rerun()
