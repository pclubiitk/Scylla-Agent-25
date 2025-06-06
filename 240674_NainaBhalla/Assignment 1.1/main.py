from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq()
temp=float(input("Enter temperature (default 2): "))
if not temp:
    temp = 2
question = input("Ask question: ")

completion = client.chat.completions.create(
    model="gemma2-9b-it",
    messages=[
      {
        "role": "user",
        "content": question
      }
    ],
    temperature=temp,
    max_completion_tokens=2048,
    #top_p=1,
    stream=True,
    stop=None,
)

for chunk in completion:
    print(chunk.choices[0].delta.content or "", end="")
