
#  Cell 1: Import libraries
import os
import cv2
import base64
​
#  Cell 1: Import libraries
import os
import base64
import google.generativeai as genai
​
#  Cell 2: Configuration
VIDEO_PATH = "/home/realtimeidns/Downloads/llmvand.mp4"
GEMINI_API_KEY = os.getenv("AIzaSyBTIbICpwJKEcyqSc0fydk8hl1BR6qzkRI") or "AIzaSyBTIbICpwJKEcyqSc0fydk8hl1BR6qzkRI"
​
# Set up Gemini
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")
​
#  Cell 3: Read and encode video
with open(VIDEO_PATH, "rb") as f:
    video_data = f.read()
    video_b64 = base64.b64encode(video_data).decode("utf-8")
​
#  Cell 4: Define prompt
prompt = [
    {
        "inline_data": {
            "mime_type": "video/mp4",
            "data": video_b64
        }
    },
    "You are a neighborhood safety assistant. Does this video show someone damaging a car—scratching, kicking, or breaking it? If yes, respond with: 'Yes: [action]'. If not, say 'No: [reason]'."
]
​
#  Cell 5: Send to Gemini
try:
    response = gemini_model.generate_content(prompt)
    print(f" Gemini Response:\n{response.text}\n")
except Exception as e:
    print(f" Gemini API error: {e}\n")
​
'''
/home/realtimeidns/genai-vandalism-env/lib/python3.12/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
  from .autonotebook import tqdm as notebook_tqdm
🤖 Gemini Response:
Yes: breaking 
'''
