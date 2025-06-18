from llama_index.llms.groq import Groq
from llama_index.core.llms import MessageRole, ChatMessage
from llama_index.core.tools import FunctionTool
from typing import NewType
import random
import os
from dotenv import load_dotenv
load_dotenv()
GROQ_API_KEY = os.environ["GROQ_API_KEY"]

llm = Groq(
    model="llama3-70b-8192",
    api_key=GROQ_API_KEY
)


handle = llm.stream_complete("Virat Kohli is ")
for token in handle:
    print(token.delta, end="", flush=True)
messages = [
    ChatMessage(role=MessageRole.SYSTEM,
                content="You are a helpful assistant."),
    ChatMessage(role=MessageRole.USER, content="Tell me a joke.")
]
chat_response = llm.chat(messages)
print(chat_response.message.content)


Song = NewType('Song', dict)


def generate_song() -> Song:
    songs = [
        {"name": "Blinding Lights", "artist": "The Weeknd"},
        {"name": "Viva La Vida", "artist": "Coldplay"},
        {"name": "Shape of You", "artist": "Ed Sheeran"}
    ]
    return random.choice(songs)


tool = FunctionTool.from_defaults(fn=generate_song)
llm2 = Groq(model="llama3-70b-8192", api_key=GROQ_API_KEY)
response = llm2.predict_and_call(
    [tool],
    "Pick a random song for me",
)
print(str(response))
