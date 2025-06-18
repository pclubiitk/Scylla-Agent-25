import os
!pip install python-dotenv
from dotenv import load_dotenv
import google.generativeai as genai
load_dotenv('API1.env')
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)
query = input("Enter query:")
model=genai.GenerativeModel('gemini-1.5-flash')
response=model.generate_content(query)

print(response.text)
