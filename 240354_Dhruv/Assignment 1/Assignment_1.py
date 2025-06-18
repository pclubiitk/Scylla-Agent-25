from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq()
temp=float(input("Enter temp required: "))

if not temp:
    temp = 0.1
else:
    if temp > 1:
        dec = input("Temp too high, might halucinate. Proceed? y/n: ")
        if dec.lower() != "y":
            temp = 0.1

while True:
    
    question = input("Ask question (or type 'exit' to quit): ")
    if question.lower().strip() == "exit":
        break

    # Call Groq API
    completion = client.chat.completions.create(
        model="gemma2-9b-it",
        messages=[
            {"role": "system", "content": "You are a sarcastic but accurate assistant."},
            {"role": "user", "content": question}
        ],
        temperature=temp,
        max_completion_tokens=2048,
        stream=True,
    )

    print("Answer:", end=" ")
    for chunk in completion:
        print(chunk.choices[0].delta.content or "", end="")
    print("\n")