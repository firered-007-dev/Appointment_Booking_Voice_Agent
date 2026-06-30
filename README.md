# 📞 AI Corporate Secretary Call Center (Tool-Calling Agent)

An intelligent, low-latency automated corporate receptionist and appointment booking platform. The system leverages advanced **Function Calling (Tool Use)** to interact with an operational calendar database, managing scheduling flows directly from a browser interface over semantic voice input.

---

## 🏗️ Technical Architecture

The platform transitions legacy native hardware dependencies into a decoupled, web-native application structure optimized for cloud server execution:

* **Deterministic Tool Execution:** Features a strict Python native calendar schema (`book_appointment`) directly exposed as an actionable tool to the model layer. 
* **Temporal Context Injection:** Dynamically evaluates real-time timestamps on every request, allowing the language model to parse relative user dates (e.g., *"tomorrow at 3 PM"* or *"next Monday"*) into explicit `YYYY-MM-DD` and `HH:MM` arguments.
* **Orchestration Layer:** Built entirely on **FastAPI**, handling multi-turn asynchronous routing payloads with structured input validation.
* **LLM Reasoning & Synthesis:** Employs the **Google GenAI SDK** targeting `gemini-2.5-flash` with dual-pass execution—first checking for intent tool triggers, and subsequently generating elegant, natural conversational feedback.
* **Zero-Cloud-Cost Client Processing:** Leverages browser-native HTML5 Web Speech APIs (`webkitSpeechRecognition` and `speechSynthesis`) on the client side. This eliminates server-side processing latency and avoids pay-per-second API gateway billing for audio transcription and synthesis.

---

## 🛠️ Tech Stack

| Layer | Technology |
| :--- | :--- |
| **Core Framework** | FastAPI, Uvicorn, Python 3.10+ |
| **Generative AI Platform** | Google Gemini 2.5 Flash Engine (with Function Calling / Tool Use) |
| **Data Layer** | Ephemeral Global State Dictionary Mapping (In-Memory DB) |
| **Client Frontend** | HTML5 Web Audio & Web Speech Recognition Infrastructure (Vanilla JS/HTML5) |

---

## 🔄 System Workflow Steps

The application processes semantic audio input and maps it to deterministic system states using a 5-step pipelined execution model:
