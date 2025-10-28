# Vision-Prompt Glasses Prototype

## Overview

This project explores a multimodal human-AI interface built around natural visual framing and voice input. The goal is to allow users to “ask questions about what they see” by framing an image with their hands, holding it steady, and speaking a query.

In the final version, two cameras will be mounted on smart-glasses hardware. For now, the prototype runs on a single laptop webcam and demonstrates the core loop:

1. Detect hands and landmarks with MediaPipe.
2. Identify a “frame” gesture formed by both hands (index + thumb tips).
3. If the gesture is held for more than 2 seconds, crop the image inside the hand-frame.
4. Encode the cropped region as base64 and send it—along with a voice or text prompt—to the OpenAI API.
5. Display or speak the model’s response.

Future versions will include real-time voice prompts, gesture-based micro-agent switching, and on-device deployment.

---

## Setup

### For Conda Environment (Recommended)
```bash
cd /Users/justinhou/Development/FPCV-Final-Project

# Activate your conda environment
conda activate cv

# Install dependencies
pip3 install -r requirements.txt

# Run the application
python main.py
# OR use the convenience script
./run.sh
```

### For System Python
```bash
cd /Users/justinhou/Development/FPCV-Final-Project

# Quick setup (if needed)
./setup.sh

# Run the application
python3 main.py
```

Keyboard Controls:
q - Quit
r - Reset gesture detection
s - Select cached snapshot
t - Test OpenAI connection

## Demo
python demo_hand_detection.py
Hand Detection Demo - Enhanced Visualization
==================================================
This demo shows enhanced hand landmarks with:
• Green circles: Finger tips
• Yellow circles: PIP joints (middle finger joints)
• Red squares: MCP joints (knuckles)
• Magenta circle: Wrist
• T/I labels: Thumb and Index finger tips

Press 'q' to quit, 'r' to reset

## Goals

### Phase 1 – Laptop Prototype

* Use [MediaPipe Hand Landmarker (Python)](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python) for real-time hand detection.
* Detect the “hand-frame” gesture (left + right index/thumb forming a rectangle).
* Confirm gesture stability for ≥ 2 seconds.
* Extract the framed region via OpenCV perspective transform.
* Encode image as base64 and send to OpenAI’s Vision API with a text/voice query.
* Return the response in console or simple UI.

### Phase 2 – Wearable Prototype

* Integrate with smart-glasses SDK (Meta / Ray-Ban / open-source camera rig).
* Determine dominant-eye camera for capture alignment.
* Add microphone input → speech-to-text → query pipeline.
* Use onboard speaker for AI responses.
* Support gesture-based micro-agents, e.g.:

  * Victory sign (held 2 s) → switch to “Save Image to Drive” agent.
  * Other gestures → contextual tasks (“Summarize what I’m seeing”, “Translate text”, etc.).

---

## System Architecture (Prototype)

```
[ Laptop Camera ]
        ↓
[ MediaPipe Hand Landmarker ]
        ↓
Detect landmarks → Identify frame gesture
        ↓
If held > 2s → Crop framed region (OpenCV)
        ↓
Convert to Base64
        ↓
[ OpenAI API ]
    ├─ Image + Prompt → Vision/Chat model
    └─ Response → Console / Voice
```

---

## Tech Stack

| Component               | Library / API                                            |
| ----------------------- | -------------------------------------------------------- |
| Hand / finger detection | [MediaPipe](https://github.com/google-ai-edge/mediapipe) |
| Image processing        | OpenCV, NumPy                                            |
| Voice input (optional)  | SpeechRecognition / Whisper                              |
| AI model                | OpenAI API (GPT-4o / GPT-4-Vision)                       |
| Backend / Orchestration | Python (FastAPI / simple script)                         |
| Future hardware         | Meta-like dual-camera glasses                            |

---

## Quick Start (Prototype)

1. Clone repo:

   ```bash
   git clone https://github.com/<your-username>/vision-prompt-glasses.git
   cd vision-prompt-glasses
   ```

2. Install dependencies:

   ```bash
   pip install mediapipe opencv-python openai numpy
   ```

3. Set up environment:

   ```bash
   export OPENAI_API_KEY="sk-..."
   ```

4. Run the prototype:

   ```bash
   python main.py
   ```

5. Use the app:

   * Place both hands in front of the webcam forming a rectangle with index + thumb.
   * Hold steady for 2 seconds.
   * Speak or type your question (e.g., "What kind of object is this?").
   * The program captures the framed region and sends it to the model.

---

## Future Directions

| Feature                   | Description                                                    |
| ------------------------- | -------------------------------------------------------------- |
| Dominant Eye Calibration  | Determine which camera aligns with user’s viewpoint.           |
| On-Device Inference       | Run MediaPipe and voice processing on embedded hardware.       |
| Gesture-Controlled Agents | Extend to pre-set micro-agents triggered by gestures.          |
| Contextual Memory         | Cache previous frames and model outputs for follow-up queries. |
| Privacy Layer             | Local preprocessing and selective cloud upload.                |

---

## Repository Structure (Proposed)

```
vision-prompt-glasses/
│
├── main.py               # Prototype entry point
├── mediapipe_utils.py    # Hand detection + landmark helpers
├── frame_detector.py     # Logic to detect and verify hand frame
├── crop_utils.py         # Perspective crop + base64 encode
├── openai_client.py      # API call + prompt management
├── voice_input.py        # (Optional) speech capture + transcription
├── config/
│   └── settings.yaml
├── README.md
└── requirements.txt
```

---

## Contributing

* Fork and clone the repo.
* Create a feature branch.
* Submit pull requests with concise descriptions.

---

## License

MIT License

---

## Vision

> "Look, frame, and ask."
> A natural interface between human perception and AI understanding — enabling hands-free, voice-driven reasoning about the world around you.
