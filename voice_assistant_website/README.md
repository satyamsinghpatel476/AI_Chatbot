# Voice Assistant Website

Standalone local website for the existing `ai_robotics_assistant` project. It
adds a FastAPI backend and vanilla HTML/CSS/JavaScript frontend where an anime
AI assistant can accept voice input, show chatbot answers, read answers aloud on
button click, and display live session metrics.

This folder is independent. It does not modify evaluator files, chatbot system
files, RAG indexes, memory files, benchmark files, result files, or prompts.

## Anime Assistant Image

No uploaded image file was available in the project root during setup, so a
small placeholder image is included. Replace it with your anime assistant image
at:

```text
voice_assistant_website/frontend/assets/assistant.png
```

The website already points to that file. If the image is missing or cannot be
loaded, the page shows a styled fallback assistant.

## Install

From the project root:

```bash
pip install -r voice_assistant_website/requirements.txt
```

## Run

```bash
cd voice_assistant_website
python app.py
```

Or:

```bash
cd voice_assistant_website
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

## Ubuntu Desktop Launcher

The folder includes:

```text
voice_assistant_website/run_voice_assistant_app.sh
voice_assistant_website/stop_voice_assistant.sh
voice_assistant_website/VoiceAssistant.desktop
```

To enable the launcher:

```bash
chmod +x ~/ai_robotics_assistant/voice_assistant_website/run_voice_assistant_app.sh
chmod +x ~/ai_robotics_assistant/voice_assistant_website/stop_voice_assistant.sh
chmod +x ~/ai_robotics_assistant/voice_assistant_website/VoiceAssistant.desktop
cp ~/ai_robotics_assistant/voice_assistant_website/VoiceAssistant.desktop ~/Desktop/
```

Then right-click the desktop icon and choose **Allow Launching**.

Double-click **Local Multi-Domain Assistant** from the Desktop. The launcher
opens without a visible terminal, activates `~/ai_robotics_assistant/env`,
starts `voice_assistant_website/app.py` in the background, and opens Google
Chrome app mode at:

```text
http://127.0.0.1:8000
```

Closing the Chrome app window stops the backend automatically. To stop it
manually:

```bash
~/ai_robotics_assistant/voice_assistant_website/stop_voice_assistant.sh
```

## Features

- Anime assistant image with speaking animation.
- Voice input using the browser Web Speech API.
- System selector for System A, System B, System C, or all systems.
- Text responses from the selected system or all systems.
- Read and Stop buttons for every answer.
- No automatic speech playback.
- Browser text-to-speech using `speechSynthesis`.
- Live metrics dashboard modal for the current localhost session.
- Chart.js graphs for latency, response counts, accuracy, contamination,
  hallucination, and leakage.
- PDF dashboard report download.
- CSV session export.
- Temporary RAG upload for PDF, TXT, CSV, DOCX, and JSON.
- Uploaded files are stored only in `temp_uploads/` and deleted on Clear Session
  or server shutdown.
- Chat history, metrics, and temporary RAG context use memory only and reset on
  server restart.

## Notes

The existing chatbot systems are imported safely from the parent project. If a
system cannot be imported or called, the website shows a clear error card for
that system instead of modifying the research project.
