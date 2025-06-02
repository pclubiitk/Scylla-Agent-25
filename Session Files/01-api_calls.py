import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.environ['GROQ_API_KEY']

client = Groq()
completion = client.chat.completions.create(
    model="gemma2-9b-it",
    messages=[
    {
        "role": "user",
        "content": "Procedure to prepare an old fashioned"
    }
    ],
    temperature=1,
    max_completion_tokens=1024,
    top_p=1,
    stream=True,
    stop=None,
)
for chunk in completion:
    print(chunk.choices[0].delta.content or "", end="")