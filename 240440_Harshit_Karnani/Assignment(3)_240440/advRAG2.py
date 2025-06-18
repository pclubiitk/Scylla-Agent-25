from langchain_core.output_parsers import StrOutputParser
import urllib.request
import fitz  # PyMuPDF
from PIL import Image
import io
import os
from dotenv import load_dotenv

import google.generativeai as genai
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_cohere import ChatCohere, CohereEmbeddings

load_dotenv()

# Download the file
url = "https://arxiv.org/pdf/1706.03762"
filename = "attention_is_all_you_need.pdf"
urllib.request.urlretrieve(url, filename)

# print(f"Downloaded: {filename}")

text_data = []
img_data = []
with fitz.open('attention_is_all_you_need.pdf') as pdf_file:
    # Create a directory to store the images
    if not os.path.exists("extracted_images"):
        os.makedirs("extracted_images")

    # Loop through every page in the PDF
    for page_number in range(len(pdf_file)):
        page = pdf_file[page_number]

        # Get the text on page
        text = page.get_text().strip()
        text_data.append({"response": text, "name": page_number+1})
        # Get the list of images on the page
        images = page.get_images(full=True)

        # Loop through all images found on the page
        for image_index, img in enumerate(images, start=0):
            xref = img[0]  # Get the XREF of the image
            base_image = pdf_file.extract_image(xref)  # Extract the image
            image_bytes = base_image["image"]  # Get the image bytes
            image_ext = base_image["ext"]  # Get the image extension

            # Load the image using PIL and save it
            image = Image.open(io.BytesIO(image_bytes))
            image.save(
                f"extracted_images/image_{page_number+1}_{image_index+1}.{image_ext}")

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
model = genai.GenerativeModel(model_name="gemini-1.5-flash")

for img in os.listdir("extracted_images"):
    image = Image.open(f"extracted_images/{img}")
    response = model.generate_content([image, "You are an assistant tasked with summarizing tables, images and text for retrieval. \
    These summaries will be embedded and used to retrieve the raw text or table elements \
    Give a concise summary of the table or text that is well optimized for retrieval. Table or text or image:"])
    img_data.append({"response": response.text, "name": img})

# Set embeddings
embedding_model = CohereEmbeddings(model="embed-english-v3.0")

# Load the document
docs_list = [Document(page_content=text['response'], metadata={
                      "name": text['name']}) for text in text_data]
img_list = [Document(page_content=img['response'], metadata={
                     "name": img['name']}) for img in img_data]

# Split
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=400, chunk_overlap=50
)

doc_splits = text_splitter.split_documents(docs_list)
img_splits = text_splitter.split_documents(img_list)

# Add to vectorstore
vectorstore = Chroma.from_documents(
    documents=doc_splits + img_splits,  # adding the both text and image splits
    collection_name="multi_model_rag",
    embedding=embedding_model,
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={'k': 1},  # number of documents to retrieve
)

query = "What is the BLEU score of the Transformer (base model)?"
docs = retriever.invoke(query)


# Prompt
system = """You are an assistant for question-answering tasks. Answer the question based upon your knowledge. 
Use three-to-five sentences maximum and keep the answer concise."""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human",
         "Retrieved documents: \n\n <docs>{documents}</docs> \n\n User question: <question>{question}</question>"),
    ]
)

# LLM
llm = ChatCohere(model="command-r-plus", temperature=0)

# Chain
rag_chain = prompt | llm | StrOutputParser()

# Run
generation = rag_chain.invoke(
    {"documents": docs[0].page_content, "question": query})
print(generation)
