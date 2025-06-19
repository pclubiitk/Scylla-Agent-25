import os
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_index.core import SimpleDirectoryReader
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_cohere import CohereEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from enum import Enum

# Step 1: Load API keys
load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
COHERE_API_KEY = os.getenv('COHERE_API_KEY')

# Step 2: Set up embeddings
embedding_model = CohereEmbeddings(model="embed-english-v3.0")

# Step 3: Load articles
from langchain_community.document_loaders import PyMuPDFLoader

loader = PyMuPDFLoader("D:/vs_code_python/SCYLLAAGENT/final_draft.pdf")
docs = loader.load()  # Returns LangChain documents with page_content and metadata
docs_list=docs

# Step 4: Split documents
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=300,
    chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)
doc_splits = [doc for doc in doc_splits if doc.page_content.strip()]

print(f"doc_splits count: {len(doc_splits)}")

# Step 5: Build FAISS vectorstore
try:
    print("Creating FAISS vectorstore...")
    vectorstore = FAISS.from_documents(doc_splits, embedding=embedding_model)
    print("FAISS vectorstore created.")
except Exception as e:
    print("Vectorstore creation failed:", e)
    exit()

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={'k': 4})

# Step 6: Ask a question
question = "What are the different kinds of text-preprocessing methods?"
retrieved_docs = retriever.invoke(question)

# Step 7: Define structured output schema
class BinaryScore(str, Enum):
    yes = "yes"
    no = "no"

class GradeDocuments(BaseModel):
    """Binary score: Is the document relevant to the user's question?"""
    binary_score: BinaryScore = Field(
        description="Answer 'yes' if the document is relevant to the question, else 'no'"
    )


# Step 8: Setup Groq LLM
llm = ChatGroq(model="llama3-8b-8192", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeDocuments)

# Step 9: Create grading prompt
system_prompt = (
    """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
)
grade_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "Document: {document}\n\nQuestion: {question}")
])
retrieval_grader = grade_prompt | structured_llm_grader

# Step 10: Grade documents
docs_to_use = []
for i, doc in enumerate(retrieved_docs):
    print(f"\nDoc {i+1}:\n", doc.page_content[:400], "\n" + "-"*60)
    try:
        result = retrieval_grader.invoke({
            "document": doc.page_content[:1000],  # Keep input small
            "question": question
        })
        print("Grader result:", result)
        if result.binary_score == BinaryScore.yes:
            docs_to_use.append(doc)
    except Exception as e:
        print("Grading failed:", e)

# Step 11: Final output
print(f"\nRelevant docs retained: {len(docs_to_use)}")
for i, doc in enumerate(docs_to_use):
    print(f"\n-- Document {i+1} Source: {doc.metadata.get('source')}")
    print(doc.page_content[:300], "...")

from langchain_core.output_parsers import StrOutputParser

#Prompt
system = """You are an assistant for question-answering tasks. Answer the question based upon your knowledge. 
Use three-to-five sentences maximum and keep the answer concise."""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system",system),
        ("human", "Retrived documents: \n\n <docs>{documents}</docs> \n\n User question: <question>{question}</question>")
    ]
)

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# Post-processing
def format_docs(docs):
    return "\n".join(f"<doc{i+1}>:\nTitle:{doc.metadata['title']}\nSource:{doc.metadata['source']}\nContent:{doc.page_content}\n</doc{i+1}>\n" for i, doc in enumerate(docs))

# Chain
rag_chain = prompt | llm | StrOutputParser()

# Run
generation = rag_chain.invoke({"documents":format_docs(docs_to_use), "question": question})
print(generation)

class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generated answer"""
    binary_score: str = Field(
        ...,
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

llm = ChatGroq(model="llama3-8b-8192", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeHallucinations)

# Prompt
system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
    Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n <facts>{documents}</facts> \n\n LLM generation: <generation>{generation}</generation>"),
    ]
)

hallucination_grader = hallucination_prompt | structured_llm_grader

response = hallucination_grader.invoke({"documents": format_docs(docs_to_use), "generation": generation})
print(response)

from typing import List
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

class HighlightDocuments(BaseModel):
    """Return specific part of a document used for answering the question"""

    id: List[str] = Field(
        ...,
        description="List of id of docs used to answers the question"
    )

    title: List[str] = Field(
        ...,
        description="List of titles used to answers the question"
    )

    source: List[str] = Field(
        ...,
        description="List of sources used to answers the question"
    )

    segment: List[str] = Field(
        ...,
        description="List of direct segements from used documents that answers the question"
    )


llm = ChatGroq(model="llama3-70b-8192", temperature=0)

parser = PydanticOutputParser(pydantic_object=HighlightDocuments)


system = """You are an advanced assistant for document search and retrieval. You are provided with the following:
1. A question.
2. A generated answer based on the question.
3. A set of documents that were referenced in generating the answer.

Your task is to identify and extract the exact inline segments from the provided documents that directly correspond to the content used to 
generate the given answer. The extracted segments must be verbatim snippets from the documents, ensuring a word-for-word match with the text 
in the provided documents.

Ensure that:
- (Important) Each segment is an exact match to a part of the document and is fully contained within the document text.
- The relevance of each segment to the generated answer is clear and directly supports the answer provided.
- (Important) If you didn't used the specific document don't mention it.

Used documents: <docs>{documents}</docs> \n\n User question: <question>{question}</question> \n\n Generated answer: <answer>{generation}</answer>

<format_instruction>
{format_instructions}
</format_instruction>
"""

prompt = PromptTemplate(
    template= system,
    input_variables=["documents", "question", "generation"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# Chain
doc_lookup = prompt | llm | parser

# Run
lookup_response = doc_lookup.invoke({"documents":format_docs(docs_to_use), "question": question, "generation": generation})

for id, title, source, segment in zip(lookup_response.id, lookup_response.title, lookup_response.source, lookup_response.segment):
    print(f"ID: {id}\nTitle: {title}\nSource: {source}\nText Segment: {segment}\n")