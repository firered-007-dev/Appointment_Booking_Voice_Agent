"""
secretary_agent.py — Voice-Activated AI Secretary with Tool-Use Function Calling
Handles natural speech, extracts structural booking arrays, and runs real-time calendar updates.
Usage:
    python secretary_agent.py
"""

import io
import os
import sys
import time
from datetime import datetime, timedelta

from dotenv import load_dotenv
import numpy as np
import sounddevice as sd
from scipy.io import wavfile
import speech_recognition as sr
import pyttsx3
from google import genai
from google.genai import types

load_dotenv()

# Audio Configurations
SAMPLE_RATE = 16000
CHANNELS = 1

if not os.getenv("GEMINI_API_KEY"):
    print("[ERROR] Missing GEMINI_API_KEY in your env file.")
    sys.exit(1)

# Initialize Core API Engine
gemini_client = genai.Client()

# Initialize Local Speaker Engine
engine = pyttsx3.init()
engine.setProperty('rate', 170)

# Mock Calendar Database (Stores scheduled appointment objects)
# Format: {"YYYY-MM-DD HH:MM": "Meeting Description / Purpose"}
calendar_db = {
    f"{(datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')} 10:00": "Existing Corporate Board Sync",
    f"{(datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')} 15:00": "Pre-booked Client Interview"
}


def book_appointment(date_str: str, time_str: str, purpose: str = "General Consultation") -> str:
    """
    Core tool function exposed to the AI model.
    Checks for scheduling conflicts and updates the local calendar database store.
    """
    print(f"\n[TOOL EXECUTION] Tool triggered by AI. Checking slots for Date: {date_str}, Time: {time_str}...")
    
    # Standardize tracking strings
    slot_key = f"{date_str.strip()} {time_str.strip()}"
    
    # Conflict check
    if slot_key in calendar_db:
        return f"REJECTION: The slot on {date_str} at {time_str} is already booked for '{calendar_db[slot_key]}'. Please ask the user to pick another time."
    
    # Book the slot
    calendar_db[slot_key] = purpose
    print(f"🗓️ [CALENDAR UPDATED] Booked successfully: {slot_key} -> '{purpose}'")
    
    return f"SUCCESS: The appointment has been secured on {date_str} at {time_str} for '{purpose}'. Sending digital confirmation card now."


def record_microphone(duration=5):
    """Captures voice audio signals via sounddevice hardware layer."""
    print(f"\n🎙️  [Secretary Listening...] Speak your booking request clear ({duration}s)...")
    audio_data = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='int16')
    sd.wait()
    print("🛑 [Stopped Listening] Decoding voice stream...")
    
    wav_buffer = io.BytesIO()
    wavfile.write(wav_buffer, SAMPLE_RATE, audio_data)
    wav_buffer.seek(0)
    return wav_buffer


def transcribe_audio_free(audio_buffer):
    """Converts audio blocks directly to text strings completely free."""
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_buffer) as source:
            audio_data = recognizer.record(source)
        user_text = recognizer.recognize_google(audio_data)
        print(f" 🗣️  You said: \"{user_text}\"")
        return user_text
    except Exception:
        print("[WARN] Could not parse speech cleanly.")
        return ""


def run_secretary_agent_loop(user_query: str):
    """
    Uses Gemini Function Calling / Tool Execution to parse dates,
    run calculations, and execute calendar management logic natively.
    """
    print("[1/3] Secretary processing request intent...")
    
    current_time_context = (
        f"The current real-time timestamp is: {datetime.now().strftime('%A, %B %d, %Y at %I:%M %p')}. "
        "Use this precise context to map relative time phrasing (e.g., 'tomorrow', 'next Monday', '3 PM') to exact YYYY-MM-DD formats."
    )
    
    system_instruction = (
        "You are an expert corporate executive AI secretary. Your goal is to help users manage their schedules. "
        "If a user wants to book, reschedule, or check an appointment, you MUST call the provided tool: 'book_appointment'. "
        "Do not guess the dates—use the real-time context provided to accurately compute the YYYY-MM-DD string. "
        "Always respond politely, professionally, and keep your verbal confirmation down to 1-2 natural sentences maximum."
    )
    
    try:
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=f"{current_time_context}\nUser Request: {user_query}",
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                tools=[book_appointment]  # Register the tool block
            )
        )
        
        # Check if the model decided it needs to execute the book_appointment function
        if response.function_calls:
            print("[2/3] Model matched criteria. Dispatching Python tool function call...")
            for call in response.function_calls:
                if call.name == "book_appointment":
                    # Extract arguments computed by the LLM
                    args = call.args
                    date_val = args.get("date_str")
                    time_val = args.get("time_str")
                    purpose_val = args.get("purpose", "General Consultation")
                    
                    # Execute the actual local function block
                    tool_result = book_appointment(
                        date_str=str(date_val), 
                        time_str=str(time_val), 
                        purpose=str(purpose_val)
                    )
                    
                    # Feed the result back to the model so it can formulate a natural response
                    follow_up_response = gemini_client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=[
                            f"User Query: {user_query}",
                            f"Tool execution result summary: {tool_result}. Formulate your short vocal verbal summary answer now."
                        ],
                        config=types.GenerateContentConfig(system_instruction=system_instruction)
                    )
                    ai_text = follow_up_response.text.strip()
                    return ai_text
        
        # If no tool execution was triggered, return standard text feedback
        return response.text.strip() if response.text else "I can help you book an appointment. What time works for you?"
        
    except Exception as e:
        print(f"[AGENT CORE ERROR] Agent execution failure: {e}")
        return "I encountered a processing error handling your calendar data."


def speak_output_offline(text: str):
    """Speaks the response sentence out loud through local speakers."""
    print(f" 🤖 AI Secretary: \"{text}\"")
    print("[3/3] Speaking confirmation...")
    engine.say(text)
    engine.runAndWait()


def main():
    print("============================================================")
    print("      AI SECRETARY CALL CENTER ONLINE (FUNCTION CALLING)    ")
    print("============================================================")
    print(f"[INFO] Reference Baseline Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    print("[INFO] Pre-existing Calendar Schedule:")
    for slot, desc in calendar_db.items():
        print(f"  • {slot} -> [{desc}]")
    print("============================================================")
    
    try:
        while True:
            # Capture sound bytes
            audio_buffer = record_microphone(duration=5)
            
            # STT Processing
            user_text = transcribe_audio_free(audio_buffer)
            if not user_text or len(user_text) < 3:
                continue
            
            # Run Autonomous Agent Loop with Tool Integration
            ai_confirmation = run_secretary_agent_loop(user_text)
            
            # Speak Response Summary aloud
            speak_output_offline(ai_confirmation)
            
            print("\n🔄 Secretary ready for next user routing request...")
            time.sleep(1.0)
            
    except KeyboardInterrupt:
        print("\n[INFO] Powering down AI Secretary workstation securely. Goodbye!")


if __name__ == "__main__":
    main()