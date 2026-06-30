import os
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

app = FastAPI(title="AI Secretary Call Center")

if not os.getenv("GEMINI_API_KEY"):
    raise RuntimeError("Missing GEMINI_API_KEY environment variable.")

# Initialize Gemini Client
gemini_client = genai.Client()

# Global Mock Calendar Database (Resets when server restarts)
calendar_db = {
    f"{(datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')} 10:00": "Existing Corporate Board Sync",
    f"{(datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')} 15:00": "Pre-booked Client Interview"
}

def book_appointment(date_str: str, time_str: str, purpose: str = "General Consultation") -> str:
    """Core tool function exposed to the AI model for schedule handling."""
    slot_key = f"{date_str.strip()} {time_str.strip()}"
    
    if slot_key in calendar_db:
        return f"REJECTION: The slot on {date_str} at {time_str} is already booked for '{calendar_db[slot_key]}'. Please ask the user to pick another time."
    
    calendar_db[slot_key] = purpose
    print(f"🗓️ [CALENDAR UPDATED] Booked successfully: {slot_key} -> '{purpose}'")
    return f"SUCCESS: The appointment has been secured on {date_str} at {time_str} for '{purpose}'."

class VoiceRequest(BaseModel):
    text: str

@app.post("/api/secretary")
async def secretary_endpoint(payload: VoiceRequest):
    user_query = payload.text.strip()
    if not user_query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

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
                tools=[book_appointment]
            )
        )
        
        # Process Function Calling / Tool Triggering
        if response.function_calls:
            for call in response.function_calls:
                if call.name == "book_appointment":
                    args = call.args
                    tool_result = book_appointment(
                        date_str=str(args.get("date_str")), 
                        time_str=str(args.get("time_str")), 
                        purpose=str(args.get("purpose", "General Consultation"))
                    )
                    
                    # Pass execution results back to model for a conversational voice response
                    follow_up = gemini_client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=[
                            f"User Query: {user_query}",
                            f"Tool execution result summary: {tool_result}. Formulate your short vocal verbal summary answer now."
                        ],
                        config=types.GenerateContentConfig(system_instruction=system_instruction)
                    )
                    return {"response": follow_up.text.strip()}
                    
        return {"response": response.text.strip() if response.text else "How can I assist with your scheduling needs today?"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/slots")
async def get_slots():
    """Endpoint to check active bookings."""
    return calendar_db

# HTML UI Layout
HTML_CONTENT = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Corporate Secretary</title>
    <style>
        body { font-family: 'Segoe UI', sans-serif; text-align: center; background: #0f172a; color: #f8fafc; padding-top: 50px; }
        .box { max-width: 600px; margin: 0 auto; background: #1e293b; padding: 30px; border-radius: 12px; }
        button { padding: 12px 24px; font-size: 16px; cursor: pointer; border-radius: 8px; border: none; background: #10b981; color: white; font-weight: bold; }
        button:hover { background: #059669; }
        #output { margin-top: 25px; font-size: 18px; color: #34d399; }
    </style>
</head>
<body>
    <div class="box">
        <h1>💼 AI Corporate Secretary Call Center</h1>
        <p>Click below to book or check appointments via voice:</p>
        <button id="talkBtn">Speak to Secretary</button>
        <p id="status">Ready.</p>
        <div id="output"></div>
    </div>
    <script>
        const talkBtn = document.getElementById('talkBtn');
        const status = document.getElementById('status');
        const output = document.getElementById('output');

        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (SpeechRecognition) {
            const recognition = new SpeechRecognition();
            recognition.lang = 'en-US';

            talkBtn.onclick = () => { recognition.start(); status.textContent = "Listening to your schedule request..."; };

            recognition.onresult = async (event) => {
                const speechText = event.results[0][0].transcript;
                status.textContent = `Analyzing input: "${speechText}"...`;
                
                const res = await fetch('/api/secretary', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: speechText })
                });
                const data = await res.json();
                
                output.textContent = "🤖 Secretary: " + data.response;
                status.textContent = "Speaking confirmation...";
                
                const utterance = new SpeechSynthesisUtterance(data.response);
                window.speechSynthesis.speak(utterance);
                utterance.onend = () => { status.textContent = "Ready."; };
            };
        } else {
            status.textContent = "Browser voice functionality missing.";
        }
    </script>
</body>
</html>
"""

@app.get("/")
async def root():
    return HTMLResponse(HTML_CONTENT)