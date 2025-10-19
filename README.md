## Collaborators: 
Tevin Kim, Justin Hou, Lucia Chen

## Goal: Detect and report high-risk home disasters in real time, so action can be taken before damage or harm escalates.

## Use Cases:
Kitchen fire — detect visible flames or smoke, especially when no human is nearby
Elder fall or heart attack — detect sudden collapse + prolonged inactivity
Baby safety — detect baby leaving crib or entering unsafe zones
Intrusion — detect unexpected human presence during off-hours or in restricted areas

## Computer Vision Components:
Object/person detection — YOLOv8 or similar
Pose estimation — OpenPose or Mediapipe for fall/body posture tracking
Zone/boundary violation — frame exit logic or ROI masking
Temporal behavior logic — inactivity timers, motion tracking
Fire/smoke detection — trained CNN on fire datasets (e.g., FIRESENSE)

## MCP Server:
Converts CV events into natural language alerts
Sends alerts via:
Websocket / TCP stream
Webhook (for SMS, Telegram, etc.)
Optional in-home text-to-speech
Sample alert:
“Fire detected near stovetop. No person in kitchen for 3+ minutes.”
“Elderly person has collapsed in hallway. No movement for 60 seconds.”

## Optional UI Interface:
Real-time alert feed with timestamps and video clip previews
Severity color-coding (e.g., red = critical)
Manual feedback (confirm or dismiss alert)

## Outcome: 
Build a working demo that combines CV-based detection with a language-based alert system to proactively flag and report home emergencies as they unfold.





Dataset: 
Fall:
https://fenix.ur.edu.pl/mkepski/ds/uf.html
https://www.kaggle.com/datasets/uttejkumarkandagatla/fall-detection-dataset
Fire:
https://www.firesense.eu/
Home Intrusions:
https://www.kaggle.com/datasets/mintumovi/residential-activity-capture-datasetracd
https://universe.roboflow.com/intrusion-detection-xvyt4/home-intrusion-ai
Models:
YOLOv8 – Fast, accurate object/person detection ideal for real-time home environments and supports many relevant classes (people, fire sources, etc.).
Mediapipe – Lightweight pose estimation for detecting falls or posture changes, easily runs on CPU and integrates with video pipelines.
CNN-based Fire Detection (e.g., FlameNet) – Specialized in recognizing flame/smoke patterns in RGB video, outperforming generic detectors in fire scenarios.
DeepSORT – Robust multi-object tracker that maintains identity across frames, essential for tracking baby movement or intruders.
Custom Temporal Rule Engine – Allows simple, interpretable logic (e.g., “no movement for 60s”) to convert raw model outputs into actionable events.


Resources:
OpenAI API credits

Midterm Checkpoint To-Do’s
Train classifier on fire (and maybe fall) detection
Finalize datasets and base models
Finetune via Optuna etc.
Figma make simple wireframes
Create UI
