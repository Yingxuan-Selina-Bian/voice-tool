# Real-Time Audio Processing and Semantic Ordering

This application captures audio in real-time, transcribes it using OpenAI's Whisper model, and then semantically reorders and summarizes the content using GPT-4.

## Prerequisites

- Python 3.9 or higher
- OpenAI API key
- Microphone access

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set your OpenAI API key as an environment variable:
```bash
# On Windows (PowerShell):
$env:OPENAI_API_KEY="your-api-key-here"

# On Windows (Command Prompt):
set OPENAI_API_KEY=your-api-key-here

# On Unix/Linux:
export OPENAI_API_KEY="your-api-key-here"
```

## Usage

### Command Line Interface
Run the application:
```bash
python audio_processor.py
```

### Streamlit Web Interface
Run the Streamlit app:
```bash
streamlit run app.py
```

The Streamlit interface provides:
- Start/Stop recording buttons
- Real-time transcription display
- Adjustable time window for grouping segments (1-30 minutes)
- One-click semantic reordering and summarization
- Expandable groups with detailed segments and summaries

## Features

- Real-time audio capture in 10-second chunks
- Automatic transcription using Whisper
- Semantic reordering of transcript segments
- Time-based grouping of segments
- Automatic summarization of grouped segments
- User-friendly web interface with Streamlit

## Notes

- The application uses the "small" Whisper model by default. You can modify the code to use other models if needed.
- Audio chunks are temporarily saved as WAV files and automatically deleted after processing.
- The application processes segments in batches of 50 to optimize API usage and processing time.
- The Streamlit interface auto-refreshes every 5 seconds while recording to show new transcriptions. 