import os
from dotenv import load_dotenv
from groq import Groq
load_dotenv()
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
client = Groq()
chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Complete the story: One day Little Red Riding hood went into forest",
        }
    ],
    model="gemma2-9b-it",
    stream=False,
    temperature=0.5,
    max_completion_tokens=4096,
)

print(chat_completion.choices[0].message.content)
