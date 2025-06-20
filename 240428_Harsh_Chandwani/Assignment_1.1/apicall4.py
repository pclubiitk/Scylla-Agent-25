from google import generativeai as genai
from dotenv import load_dotenv
import os
load_dotenv()

genai.configure(api_key="GEMINI_API_KEY")

model = genai.GenerativeModel("gemini-1.5-flash") 

response = model.generate_content("Explain how AI works in a few words")

print(response.text)