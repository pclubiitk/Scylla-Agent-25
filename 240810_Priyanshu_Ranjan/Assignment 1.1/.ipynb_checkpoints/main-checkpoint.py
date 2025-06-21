from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY= os.environ['GROQ_API_KEY']
client = Groq()
completion = client.chat.completions.create(
    # model="meta-llama/llama-4-scout-17b-16e-instruct",
    model= "deepseek-r1-distill-llama-70b",
    messages=[
      {
        "role": "user",
        "content": "Generate a short conversation between Donald Trump and Che Guevara"
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
