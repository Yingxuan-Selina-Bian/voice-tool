import sounddevice as sd
import numpy as np
import queue
import threading
import time
import wave
import whisper
import openai
from typing import List, Dict
import os
from datetime import datetime

# Global variables
q = queue.Queue()
buffer: List[Dict] = []
CHUNK_DURATION = 30  # seconds
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 1024

def callback(indata, frames, time, status):
    """Callback function for audio stream"""
    if status:
        print(f"Status: {status}")
    q.put(indata.copy())

def save_audio_chunk(chunk: np.ndarray, filename: str):
    """Save audio chunk to WAV file"""
    wf = wave.open(filename, "wb")
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(2)  # 2 bytes per sample
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes(chunk.tobytes())
    wf.close()

def process_audio_chunk(filename: str):
    """Process audio chunk using Whisper"""
    try:
        result = model.transcribe(filename, language="en")
        segments = [{"text": s["text"], "start": s["start"]} for s in result["segments"]]
        buffer.extend(segments)
        print(f"Transcribed: {len(segments)} segments")
    except Exception as e:
        print(f"Error processing audio chunk: {e}")

def recorder():
    """Background thread for recording audio"""
    chunk_counter = 0
    while True:
        try:
            # Collect chunks for CHUNK_DURATION seconds
            chunks = []
            for _ in range(SAMPLE_RATE * CHUNK_DURATION // CHUNK_SIZE):
                chunk = q.get()
                chunks.append(chunk)
            
            # Combine chunks and save
            audio_data = np.concatenate(chunks)
            filename = f"chunk_{chunk_counter}.wav"
            save_audio_chunk(audio_data, filename)
            
            # Process the chunk
            process_audio_chunk(filename)
            
            # Clean up
            os.remove(filename)
            chunk_counter += 1
            
        except Exception as e:
            print(f"Error in recorder thread: {e}")

def reorder_segments(segments: List[Dict], api_key: str) -> List[Dict]:
    """Reorder segments using GPT-4"""
    openai.api_key = api_key 
    
    # Create prompt
    prompt = "We have these transcript snippets from a meeting:\n"
    for i, seg in enumerate(segments, 1):
        prompt += f"{i}. [{seg['start']:.2f}s] {seg['text']}\n"
    prompt += "\nPlease reorder them into the most coherent logical flow. Respond with the new index order (e.g., '2,1,3,...'):"
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        order = [int(i) for i in response.choices[0].message.content.split(',')]
        return [segments[i-1] for i in order]
    except Exception as e:
        print(f"Error in reordering: {e}")
        return segments

def group_segments(segments: List[Dict], time_window: int = 300) -> List[List[Dict]]:
    """Group segments by time window (default 5 minutes)"""
    groups, current_group = [], []
    base_time = segments[0]["start"] if segments else 0
    
    for seg in segments:
        if seg["start"] - base_time <= time_window:
            current_group.append(seg)
        else:
            groups.append(current_group)
            current_group = [seg]
            base_time = seg["start"]
    
    if current_group:
        groups.append(current_group)
    
    return groups

def summarize_group(group: List[Dict], api_key: str) -> str:
    """Summarize a group of segments using GPT-4"""
    openai.api_key = api_key
    
    text = " ".join(seg["text"] for seg in group)
    prompt = f"Summarize the following in 2 sentences:\n\n{text}"
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in summarization: {e}")
        return ""

def main():
    # Load Whisper model
    global model
    model = whisper.load_model("small")
    
    # Get OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set OPENAI_API_KEY environment variable")
        return
    
    # Start audio stream
    stream = sd.InputStream(
        callback=callback,
        channels=CHANNELS,
        samplerate=SAMPLE_RATE,
        blocksize=CHUNK_SIZE
    )
    
    # Start recording thread
    record_thread = threading.Thread(target=recorder, daemon=True)
    record_thread.start()
    
    print("Recording started. Press Ctrl+C to stop.")
    
    try:
        with stream:
            while True:
                time.sleep(1)
                if len(buffer) >= 50:  # Process every 50 segments
                    print("\nProcessing segments...")
                    to_reorder = buffer[-50:]
                    reordered = reorder_segments(to_reorder, api_key)
                    groups = group_segments(reordered)
                    
                    print("\nSummaries:")
                    for i, group in enumerate(groups, 1):
                        summary = summarize_group(group, api_key)
                        print(f"\nGroup {i}:")
                        print(summary)
                    
                    buffer.clear()  # Clear buffer after processing
                
    except KeyboardInterrupt:
        print("\nStopping recording...")
    finally:
        stream.stop()
        stream.close()

if __name__ == "__main__":
    main() 