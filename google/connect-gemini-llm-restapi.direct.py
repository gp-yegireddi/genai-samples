import requests
import os
import json
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Set the API URL for the Gemini model
api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"

# Define the headers for the request
headers = {
    "x-goog-api-key": GOOGLE_API_KEY,
    "Content-Type": "application/json"
}

# Define the payload (same as the -d argument in curl)
payload = {
    "contents": [
        {
            "parts": [
                {
                    "text": "Explain how AI works in a few words"
                }
            ]
        }
    ]
}

# Make the POST request to the Gemini API
response = requests.post(api_url, headers=headers, json=payload)

# Check if the request was successful
if response.status_code == 200:
    print("Response:")
    print(json.dumps(response.json(), indent=2))
else:
    print(f"Error {response.status_code}: {response.text}")
