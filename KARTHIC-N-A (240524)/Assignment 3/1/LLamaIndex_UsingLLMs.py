"""
3 main parts - 
1) Building a RAG pipeline
2) Building an Agent
3) Building Workflows
"""

#------USING LLMS------#
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.llms import ChatMessage, TextBlock, ImageBlock
from llama_index.core.tools import FunctionTool
from pydantic import *
import os
import dotenv

dotenv.load_dotenv()
def usingllms():
    #COMPLETION
    GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
    llm = GoogleGenAI(model="gemini-2.0-flash", api_key = GOOGLE_API_KEY)
    response = llm.complete("Who is William Shakespeare? ")
    print(response, end="\n\nCAPTAIN JACK SPARROW:\n")
    #CHATTING
    messages = [
        ChatMessage(role="system", content="You are Captain Jack Sparrow"),
        ChatMessage(role="user", content="Tell me us the tale of how you defeated Davy Jones in 'At the World's End'.")
    ]
    response = llm.chat(messages)
    print(response, end="\n\nDESCRIPTION OF PICTURE:\n")
    #IMAGE
    messages = [
    ChatMessage(role="user", blocks=[ImageBlock(path="aragornimage.jpeg"),TextBlock(text="Explain the character in the image")])
    ]
    response = llm.chat(messages)
    print(response, end="\n\nTOOL CALLING:\n")
    #TOOL CALLING
    class Song(BaseModel):
        name: str
        artist: str
    def get_song_details(name: str, artist: str) -> Song:
        """
        Get details for a specific song given its name and artist.
        Args:
            name (str): The title of the song.
            artist (str): The name of the artist or band.
        """
        return Song(name=name, artist=artist)
    tools = [FunctionTool.from_defaults(fn=get_song_details)]   
    prompt = "Find a famous song composed by A.R. Rahman and get me its details."
    response = llm.predict_and_call(tools, prompt) #Predict and Call function!
    print(str(response))

def main():
    os.system("clear")
    usingllms()

main()
