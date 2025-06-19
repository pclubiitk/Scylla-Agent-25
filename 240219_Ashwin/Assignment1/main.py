import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit']:
        break

    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": user_input}],
            model="llama3-8b-8192",
            max_tokens=1024,
            temperature=0.7,
        )
        print("\n🤖 Assistant:", response.choices[0].message.content)
    except Exception as e:
        print("❌ Error:", e)
