import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.environ['GROQ_API_KEY']

client = Groq()
completion = client.chat.completions.create(
    #model="gemma2-9b-it",
    #model="llama3-8b-8192",
    model="llama-3.1-8b-instant",
    #model="deepseek-r1-distill-llama-70b",
    #model="deepSeek-r1-distill-qwen-32b",
    messages=[
    {
        "role": "user",
        "content" : "How to make IIT Kanpur"
    }
    ],
    temperature=0.1,
    max_completion_tokens=1024,
    top_p=1,
    stream=True,
    stop=None,
)
for chunk in completion:
    print(chunk.choices[0].delta.content or "", end="")