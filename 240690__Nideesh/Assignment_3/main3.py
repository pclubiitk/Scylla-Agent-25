import os
import numpy as np
from typing import List
from scipy.stats import beta
import matplotlib.pyplot as plt
import cohere

from google.colab import userdata
CO_API_KEY = userdata.get('CO_API_KEY')

from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_into_chunks(text: str, chunk_size: int):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0, length_function=len)
    texts = text_splitter.create_documents([text])
    chunks = [text.page_content for text in texts]
    return chunks

def transform(x: float):
    a, b = 0.4, 0.4  # These can be adjusted to change the distribution shape
    return beta.cdf(x, a, b)

def rerank_chunks(query: str, chunks: List[str]):
    model = "rerank-english-v3.0"
    client = cohere.Client(api_key=CO_API_KEY)
    decay_rate = 30

    reranked_results = client.rerank(model=model, query=query, documents=chunks)
    results = reranked_results.results
    reranked_indices = [result.index for result in results]
    reranked_similarity_scores = [result.relevance_score for result in results]

    similarity_scores = [0] * len(chunks)
    chunk_values = [0] * len(chunks)
    for i, index in enumerate(reranked_indices):
        absolute_relevance_value = transform(reranked_similarity_scores[i])
        similarity_scores[index] = absolute_relevance_value
        chunk_values[index] = np.exp(-i/decay_rate)*absolute_relevance_value
    return similarity_scores, chunk_values

def plot_relevance_scores(chunk_values: List[float], start_index: int = None, end_index: int = None) -> None:
    plt.figure(figsize=(12, 5))
    plt.title(f"Similarity of each chunk in the document to the search query")
    plt.ylim(0, 1)
    plt.xlabel("Chunk index")
    plt.ylabel("Query-chunk similarity")
    if start_index is None:
        start_index = 0
    if end_index is None:
        end_index = len(chunk_values)
    plt.scatter(range(start_index, end_index), chunk_values[start_index:end_index])

import os
os.makedirs('data', exist_ok=True)

# Download the PDF document used in this notebook
!wget -O data/Understanding_Climate_Change.pdf https://raw.githubusercontent.com/NirDiamant/RAG_TECHNIQUES/main/data/Understanding_Climate_Change.pdf
!wget -O data/nike_2023_annual_report.txt https://raw.githubusercontent.com/NirDiamant/RAG_TECHNIQUES/main/data/nike_2023_annual_report.txt

FILE_PATH = "data/nike_2023_annual_report.txt"

with open(FILE_PATH, 'r') as file:
    text = file.read()

chunks = split_into_chunks(text, chunk_size=800)

print (f"Split the document into {len(chunks)} chunks")

query = "Nike consolidated financial statements"

similarity_scores, chunk_values = rerank_chunks(query, chunks)
plot_relevance_scores(chunk_values)

plot_relevance_scores(chunk_values, 320, 340)

def print_document_segment(chunks: List[str], start_index: int, end_index: int):
    for i in range(start_index, end_index):
        print(f"\nChunk {i}")
        print(chunks[i])

print_document_segment(chunks, 320, 340)

def get_best_segments(relevance_values: list, max_length: int, overall_max_length: int, minimum_value: float):
    best_segments = []
    scores = []
    total_length = 0
    while total_length < overall_max_length:
        best_segment = None
        best_value = -1000
        for start in range(len(relevance_values)):
            if relevance_values[start] < 0:
                continue
            for end in range(start+1, min(start+max_length+1, len(relevance_values)+1)):
                if relevance_values[end-1] < 0:
                    continue
                if any(start < seg_end and end > seg_start for seg_start, seg_end in best_segments):
                    continue
                if total_length + end - start > overall_max_length:
                    continue

                segment_value = sum(relevance_values[start:end])
                if segment_value > best_value:
                    best_value = segment_value
                    best_segment = (start, end)

        if best_segment is None or best_value < minimum_value:
            break

        best_segments.append(best_segment)
        scores.append(best_value)
        total_length += best_segment[1] - best_segment[0]

    return best_segments, scores

import google.generativeai as genai
from google.colab import userdata

GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel('gemini-1.5-flash')

max_segment_length = 10  # Example value: maximum number of chunks in a segment
overall_max_length = 50 # Example value: maximum total number of chunks across all selected segments
minimum_relevance_value = 0.5 # Example value: minimum relevance score for a segment to be considered
best_segments, scores = get_best_segments(chunk_values, max_segment_length, overall_max_length, minimum_relevance_value)

prompt_parts = []
prompt_parts.append("Based on the following information, answer the question:\n\n")

for start, end in best_segments:
    for i in range(start, end):
        prompt_parts.append(f"Chunk {i}:\n{chunks[i]}\n\n")

prompt_parts.append(f"Question: {query}\n")
prompt_parts.append("Answer:")

formatted_prompt = "".join(prompt_parts)

response = model.generate_content(formatted_prompt)
llm_output = response.text
print("\nGemini LLM Output:")
print(llm_output)
