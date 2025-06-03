from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY= os.environ['GROQ_API_KEY']
client = Groq()
completion = client.chat.completions.create(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    messages=[
      {
        "role": "user",
        "content": "Imagine you spawn as Che Guevara at the board meeting of Morgan Stanley, what will you say"
      }
    ],
    temperature= 1.5,
    max_completion_tokens=1024,
    top_p=1,
    stream=True,
    stop=None,
)

for chunk in completion:
    print(chunk.choices[0].delta.content or "", end="")
