#  Cell 1: Import libraries
import os
import cv2
import base64

#  Cell 1: Import libraries
import os
import base64
import google.generativeai as genai

#  Cell 2: Configuration
VIDEO_PATH = "/home/realtimeidns/Downloads/CarVandalism21.mov"
GEMINI_API_KEY = os.getenv("key") or "keyM"

# Set up Gemini
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

#  Cell 3: Read and encode video
with open(VIDEO_PATH, "rb") as f:
    video_data = f.read()
    video_b64 = base64.b64encode(video_data).decode("utf-8")

#  Cell 4: Define prompt
prompt = [
    {
        "inline_data": {
            "mime_type": "video/mp4",
            "data": video_b64
        }
    },
    """
You are an AI video analyst detecting vandalism.

Please watch the entire video carefully before giving your final answer.

Classify any of the following as car vandalism:
- Kicking, scratching, hitting, or jumping on a car
- Throwing any object (like a bottle) at a car — even if it bounces off
- Any aggressive physical action directed at a parked or moving vehicle

If the action happens at the **end of the video**, still count it.

Respond with:
- Yes: [what action occurred, when]
- No: [brief reason]

Also look for subtle vandalism acts such as:
- A person dragging a key or sharp object along a car’s surface (a common act of “keying”)
- Minor but deliberate contact, especially near the car doors or sides
- Watch for suspicious hand movements as someone passes close to a car

Reminder: If someone throws an object at a car, that **counts as vandalism**, regardless of whether damage is visible. Watch closely — the key action occurs in the last few seconds
"""
]

#  Cell 5: Send to Gemini
try:
    response = gemini_model.generate_content(prompt)
    print(f" Gemini Response:\n{response.text}\n")
except Exception as e:
    print(f" Gemini API error: {e}\n")
