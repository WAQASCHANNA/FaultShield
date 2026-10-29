import os
import sys
from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI

api_key = os.environ.get("NEBIUS_API_KEY", "").strip()
base_url = "https://api.tokenfactory.nebius.com/v1"

print(f"API Key present: {bool(api_key)}")
print(f"Key preview: {api_key[:12]}...{api_key[-6:]}" if api_key else "None")
print(f"Base URL: {base_url}")

client = OpenAI(base_url=base_url, api_key=api_key, timeout=10.0)

try:
    print("Testing chat completion with model: nvidia/nemotron-3-super-120b-a12b ...")
    response = client.chat.completions.create(
        model="nvidia/nemotron-3-super-120b-a12b",
        messages=[
            {"role": "user", "content": "Ping test: respond with 'PONG'"}
        ],
        max_tokens=10,
    )
    print("SUCCESS! Model response:")
    print(response.choices[0].message.content)
except Exception as e:
    print(f"Error during API call: {e}")
