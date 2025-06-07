import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# GROQ_API_KEY = os.environ['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']

client = Groq(api_key='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')
temp = float(input('What is the temperature :'))
if temp<0 or temp>2 : 
    temp = 1
completion = client.chat.completions.create(
    # model="gemma2-9b-it"
    model = "deepseek-r1-distill-llama-70b",
    messages=[
    {
        "role": "user",
        "content": "Explain LLMs"
    }
    ],
    temperature=temp,
    max_completion_tokens=1024,
    top_p=1,
    stream=True,
    stop=None,
)
for chunk in completion:
    print(chunk.choices[0].delta.content or "", end="")
