from dotenv import load_dotenv
import os
load_dotenv('.env')
from groq import Groq
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
query = input("Ask question: ")
terminate='exit'
while query!= terminate:
 completion = client.chat.completions.create(
    model="gemma2-9b-it",
    messages=[
      {
        "role": "user",
        "content": query
      }
    ],
    temperature=0.5,
    max_completion_tokens=1024,
    top_p=1,
    stream=True,
    stop=None,
)

 for chunk in completion:
    print(chunk.choices[0].delta.content or "", end="")
 query= input("Ask question: ")