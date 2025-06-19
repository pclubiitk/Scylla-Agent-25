import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from groq import Groq
from pydantic import BaseModel
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

class RetrievalEvaluatorInput(BaseModel):
    relevance_score: float 

class QueryRewriterInput(BaseModel):
    query: str 

class KnowledgeRefinementInput(BaseModel):
    key_points: str 

class CRAG:
    def __init__(self, path, model="llama-3.1-8b-instant", max_tokens=1000, temperature=0, lower_threshold=0.3,
                 upper_threshold=0.7):
        
        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold
        from langchain_community.document_loaders import PyMuPDFLoader

        path = "Understanding_Climate_Change.pdf"
        loader = PyMuPDFLoader(path)
        pages = loader.load()
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(pages)
        texts = [doc.page_content for doc in chunks]
        embedding = CohereEmbeddings(
            cohere_api_key=os.getenv("COHERE_API_KEY"),  
            model="embed-english-v3.0"
        )
        self.vectorstore = Chroma.from_texts(texts, embedding=embedding)
        from langchain_groq import ChatGroq
        self.llm = ChatGroq(
            model_name="llama3-70b-8192",
            groq_api_key=os.environ["GROQ_API_KEY"]
        )
        self.search = DuckDuckGoSearchResults()
        
    @staticmethod
    def retrieve_documents(query, chroma_index, k=3):
        docs = chroma_index.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]
    
    def evaluate_documents(self, query, documents):
        return [self.retrieval_evaluator(query, doc) for doc in documents]
    
    def retrieval_evaluator(self, query, document):
        prompt = PromptTemplate(
            input_variables=["query", "document"],
            template="On a scale from 0 to 1, how relevant is the following document to the query? "
                     "Query: {query}\nDocument: {document}\nRelevance score:"
        )
        chain = prompt | self.llm.with_structured_output(RetrievalEvaluatorInput)
        input_variables = {"query": query, "document": document}
        result = chain.invoke(input_variables).relevance_score
        return result
        
    def perform_web_search(self, query):
        rewritten_query = self.rewrite_query(query)
        web_results = self.search.run(rewritten_query)
        web_knowledge = self.knowledge_refinement(web_results)
        sources = self.parse_search_results(web_results)
        return web_knowledge, sources

    def generate_response(self, query, knowledge, sources):
        response_prompt = PromptTemplate(
            input_variables=["query", "knowledge", "sources"],
            template="Based on the following knowledge, answer the query. "
                     "Include the sources with their links (if available) at the end of your answer:"
                     "\nQuery: {query}\nKnowledge: {knowledge}\nSources: {sources}\nAnswer:"
        )
        input_variables = {
            "query": query,
            "knowledge": knowledge,
            "sources": "\n".join([f"{title}: {link}" if link else title for title, link in sources])
        }
        response_chain = response_prompt | self.llm
        return response_chain.invoke(input_variables).content

    def run(self, query):
        print(f"\nProcessing query: {query}")
        retrieved_docs = self.retrieve_documents(query, self.vectorstore)
        eval_scores = self.evaluate_documents(query, retrieved_docs)

        print(f"\nRetrieved {len(retrieved_docs)} documents")
        print(f"Evaluation scores: {eval_scores}")

        max_score = max(eval_scores)
        sources = []

        if max_score > self.upper_threshold:
            print("\nAction: Correct - Using retrieved document")
            best_doc = retrieved_docs[eval_scores.index(max_score)]
            final_knowledge = best_doc
            sources.append(("Retrieved document", ""))
        elif max_score < self.lower_threshold:
            print("\nAction: Incorrect - Performing web search")
            final_knowledge, sources = self.perform_web_search(query)
        else:
            print("\nAction: Ambiguous - Combining retrieved document and web search")
            best_doc = retrieved_docs[eval_scores.index(max_score)]
            retrieved_knowledge = self.knowledge_refinement(best_doc)
            web_knowledge, web_sources = self.perform_web_search(query)
            final_knowledge = "\n".join(retrieved_knowledge + web_knowledge)
            sources = [("Retrieved document", "")] + web_sources

        print("\nFinal knowledge:")
        print(final_knowledge)

        print("\nSources:")
        for title, link in sources:
            print(f"{title}: {link}" if link else title)

        print("\nGenerating response...")
        response = self.generate_response(query, final_knowledge, sources)
        print("\nResponse generated")
        return response
        
if __name__ == "__main__":
    path = "Understanding_Climate_Change.pdf"
    model = "llama3-70b-8192"  
    max_tokens = 1000
    temperature = 0.0
    query = "What are the main causes of climate change?"
    lower_threshold = 0.3
    upper_threshold = 0.7
    crag = CRAG(
        path=path,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        lower_threshold=lower_threshold,
        upper_threshold=upper_threshold
    )

    response = crag.run(query)
    print(f"Query: {query}")
    print(f"Answer: {response}")
