## 0.Install Libraries
"""

!pip install sentence_transformers

!pip install chromadb

!pip install google-genai

!pip install --upgrade openai

!pip install cohere

!pip install mistralai

"""## 1. Load and Preprocess"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
import requests
import os
import json
from google import genai
import openai
from openai import OpenAI
from mistralai import Mistral
import cohere
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()

file_path = '/content/stock_news.json'

with open(file_path, 'r') as f:
    stock_data = json.load(f)


all_records = []
for company, news_list in stock_data.items():
    for news_item in news_list:
        news_item['company'] = company  # Add company name to each record
        all_records.append(news_item)

stock_news_df = pd.DataFrame(all_records)
stock_news_df

# Sanity check if ticker and company are same for all records
stock_news_df.query('ticker == company').shape

# Check for null values
print("Null values per column:")
print(stock_news_df.isnull().sum())

# Check for blank strings in columns
print("\nNumber of blank strings in each column:")
for column in stock_news_df.columns:
    num_blanks = (stock_news_df[column].astype(str).str.strip() == '').sum()
    print(f"{column}: {num_blanks}")

# Create rich document representations with news descriptions
def create_news_documents(stock_news_df):
    """Create rich document representations of news."""
    documents = []

    for idx, news in stock_news_df.iterrows():
        company = news['company']
        title = news['title']
        full_text = news['full_text']
        link = news['link']


        # Create document text - this will be embedded
        doc = f"company: {company}\n"
        if title:
            doc += f"title: {title}\n"
        if full_text:
            doc += f"full_text: {full_text}\n"
        if link:
            doc += f"link: {link}\n"

        documents.append({
            'id': str(idx),
            'content': doc,
            'metadata': {
                'title': title,
                'full_text': full_text,
                'link': link,
                'company': company
            }
        })

    return documents

# Create our document collection
news_documents = create_news_documents(stock_news_df)
print(f"Created {len(news_documents)} news documents")
print(f"Sample document:")
print(news_documents[1]['content'])

"""## 3. Create our Embeddings"""

# Initialize embedding model
# Using a smaller model for speed, but can use more powerful ones
model = SentenceTransformer('all-mpnet-base-v2')

# Initialize ChromaDB
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="my_news_index")

# chroma_client.delete_collection(name="my_news_index")

# news_documents[0]

# Process in batches to avoid memory issues
batch_size = 1000
for i in range(0, len(news_documents), batch_size):
    batch = news_documents[i:i+batch_size]

    ids = [doc['id'] for doc in batch]
    contents = [doc['content'] for doc in batch]
    metadatas = [doc['metadata'] for doc in batch]

    # Generate embeddings
    embeddings = model.encode(contents)

    # Add to collection
    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=contents,
        metadatas=metadatas
    )

print(f"Added {collection.count()} news to the vector database")

print("\nTest  vector store with a sample query...")
test_query = "how is apple stock performing"

test_embedding = model.encode(test_query).tolist()

results = collection.query(query_embeddings=[test_embedding], n_results=3)

for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
    print(f"\nResult {i+1}:")
    print(f"title: {metadata['title']}")
    print(f"full_text: {metadata['full_text']}")
    print(f"link: {metadata['link']}")

results

"""## 4. Create Retrieval

### 4.1 Basic Retrieval
"""

def query_expansion(query):
    """Expand the query for better retrieval of financial news."""
    # Simple rule-based expansions
    expansions = [
        query, # Original query
        f"Latest news about {query}",
        f"{query} stock news",
        f"{query} financial performance",
        f"{query} market impact",
    ]
    return expansions

def retrieve_news_info(query, n_results=5):
    """Retrieve relevant news information for a query."""
    # Expand query for better recall
    expanded_queries = query_expansion(query)

    all_results = []
    for expanded_query in expanded_queries:
        # Generate embedding for the query
        query_embedding = model.encode(expanded_query).tolist()

        # Retrieve relevant documents
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )

        # Add to results list
        for doc, metadata, id in zip(results['documents'][0],
                                    results['metadatas'][0],
                                    results['ids'][0]):
            all_results.append({
                'id': id,
                'document': doc,
                'metadata': metadata,
                'query': expanded_query
            })

    # Remove duplicates (same news link)
    unique_results = {}
    for result in all_results:
        if result['id'] not in unique_results:
            unique_results[result['id']] = result

    return list(unique_results.values())

retrieve_news_info("how is apple stock performing")

with open("keys.json") as f:
    keys = json.load(f)

llama_key = keys["llama_api_key"]
gemini_key = keys["gemini_api_key"]
mistral_key = keys["mistral_api_key"]
cohere_key = keys["cohere_api_key"]

# Check if have access to Gemini
client = genai.Client(api_key=gemini_key)

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=["whats your name"]
)
print(response.text)

def generate_rag_response(query, context, links):
    # Format the prompt with retrieved context and links
    prompt = f"""
    You are a knowledgeable financial news assistant with a deep understanding of companies, stock performance, and market events.

    Use the following retrieved news articles to answer the user's financial question clearly and accurately.

    User Query:
    {query}

    Retrieved News Articles:
    {context}

    Based on this information, write a helpful and well-organized answer to the user’s question.
    Summarize relevant insights across the articles — such as stock movements, earnings, leadership changes, AI updates, regulations, or market sentiment.

    Do not include source links in your answer.
    """

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[prompt],
        config={"response_mime_type": "text/plain"}
    )

    return response.text


def news_rag(user_query):
    # Step 1: Retrieve relevant documents
    results = retrieve_news_info(user_query)

    # Step 2: Format context from documents
    context = "\n\n".join([res['document'] for res in results])

    # Step 3: Extract links for reference
    links = [res['metadata']['link'] for res in results if 'link' in res['metadata']]

    # Step 4: Generate the final RAG-based answer
    response = generate_rag_response(user_query, context, links)

    # Step 5: Append source links to the response
    if links:
        response += "\n\nSources:\n" + "\n".join(links)

    return response, results

response, results = news_rag("how is apple stock performing")

print(response)

# Test financial queries
test_queries = [
    "How has Tesla's stock been performing lately?",
    "Is Microsoft making any new AI-related moves?",
    "Recent financial news about Nvidia and the semiconductor industry"
]

for query in test_queries:
    print(f"\nQuery: {query}")
    response, _ = news_rag(query)  # Ignore the results here
    print("\nResponse:")
    print(response)
    print("-" * 60)

"""### 4.2 Hyde Retrieval"""

def hyde_retrieval(query, n_results=5):
    """Use HyDE to retrieve financial news more effectively."""

    # Step 1: Generate a hypothetical financial news-style document
    hyde_prompt = f"""
    You are a financial analyst. Write a detailed news-style explanation that would best answer this user's question:

    "{query}"

    Include relevant events, companies, stock performance, financial terms, and context where appropriate.
    """

    hyde_response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[hyde_prompt],
        config={
            "response_mime_type": "text/plain"
        }
    )

    hypothetical_document = hyde_response.text

    # Step 2: Create embedding from the generated text
    hyde_embedding = model.encode(hypothetical_document).tolist()

    # Step 3: Retrieve similar documents using vector search
    results = collection.query(
        query_embeddings=[hyde_embedding],
        n_results=n_results
    )

    # Step 4: Format the results
    formatted_results = []
    for doc, metadata, id in zip(results['documents'][0],
                                 results['metadatas'][0],
                                 results['ids'][0]):
        formatted_results.append({
            'id': id,
            'document': doc,
            'metadata': metadata,
            'query': query
        })

    return formatted_results


def finance_hyde_rag(user_query):
    # Step 1: Retrieve relevant documents using HyDE
    results = hyde_retrieval(user_query, n_results=15)

    # Step 2: Format retrieved context
    context = "\n\n".join([res['document'] for res in results])

    # Step 3: Generate plain language answer (with sources appended externally)
    response = generate_rag_response(user_query, context)

    return response, results

response, results = news_rag("how is apple stock performing")

print(response)

"""### 4.3 Retrieval with Query Decomposition"""

class Query(BaseModel):
    query: str

def decompose_query(query):
    """Decompose complex financial query into simpler sub-queries."""

    decompose_prompt = f"""
    Break down the following complex financial news question into 2–3 simpler sub-questions:
    "{query}"

    The sub-questions should focus on specific companies, stock movements, events, or trends mentioned in the original query.
    Return only the sub-queries in JSON format.
    """

    decompose_response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[decompose_prompt],
        config={
            "response_mime_type": "application/json",
            "response_schema": list[Query],
        }
    )

    # Parse the response to get sub-queries
    sub_queries = json.loads(decompose_response.text)
    sub_queries = [q['query'].strip() for q in sub_queries]

    return sub_queries

complex_query = "What recent announcements have Microsoft and Tesla made, and how might these affect their stock performance?"
decompose_query(complex_query)

def retrieve_with_decomposition(query, n_results=3):
    """Retrieve relevant financial news using query decomposition for complex queries."""
    print(f"Original query: {query}")

    # Step 1: Decompose the query into simpler sub-queries
    sub_queries = decompose_query(query)
    print(f"Decomposed into: {sub_queries}")

    # Step 2: Retrieve documents for each sub-query
    all_results = []
    for sub_query in sub_queries:
        # Retrieve news results for the sub-query
        sub_results = retrieve_news_info(sub_query, n_results=n_results)
        print(f"Retrieved {len(sub_results)} results for: '{sub_query}'")
        all_results.extend(sub_results)

    # Step 3: Deduplicate results based on unique id
    unique_results = {}
    for result in all_results:
        if result['id'] not in unique_results:
            unique_results[result['id']] = result

    return list(unique_results.values())

def finance_decomposition_rag(user_query):
    # Retrieve documents with decomposition
    results = retrieve_with_decomposition(user_query, n_results=5)

    # Format context
    context = "\n\n".join([res['document'] for res in results])

    # Extract links from metadata
    links = [res['metadata'].get('link') for res in results if 'link' in res['metadata'] and res['metadata']['link']]

    # Generate response (without links inside prompt)
    response = generate_rag_response(user_query, context, links)

    # Append links explicitly if needed (optional)
    if links:
        response += "\n\nSources:\n" + "\n".join(links)

    return response, results

response, results = finance_decomposition_rag(complex_query)

print(response)

"""# Compare responses across different models and prompts"""

def call_gemini(prompt):
    from google.generativeai import GenerativeModel, configure

    configure(api_key=gemini_key)
    model = GenerativeModel("gemini-2.0-flash")
    return model.generate_content(prompt).text.strip()


# def call_openai(prompt):
#     client = OpenAI(api_key=openai_key)

#     response = client.chat.completions.create(
#         model="gpt-3.5-turbo",
#         messages=[
#             {"role": "user", "content": prompt}
#         ]
#     )

#     return response.choices[0].message.content.strip()


# def call_deepseek(prompt):
#     client = OpenAI(
#         api_key=deepseek_key,
#         base_url="https://api.deepseek.com"
#     )

#     response = client.chat.completions.create(
#         model="deepseek-chat",
#         messages=[{"role": "user", "content": prompt}]
#     )

#     return response.choices[0].message.content.strip()



def call_llama(prompt):
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=llama_key,
    )

    response = client.chat.completions.create(
        model="meta-llama/llama-3.3-70b-instruct",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content.strip()




def call_mistral(prompt):
    """Call Mistral chat model using the official SDK."""
    client = Mistral(api_key=mistral_key)

    response = client.chat.complete(
        model="mistral-small-2506",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content.strip()


def call_cohere(prompt):
    import cohere
    co = cohere.Client(cohere_key)

    response = co.chat(message=prompt, model="command-r")
    return response.text.strip()

def generate_rag_response_model_comp(query, context, model_name="gemini", prompt_template=None):
    if prompt_template is None:
        prompt_template = """You are a knowledgeable financial news assistant with expertise in analyzing company announcements, stock trends, and macroeconomic events.

User Query:
{query}

Recent News Context:
{context}

Provide a concise, helpful answer based on the above news. Avoid hallucinating facts or sources.
"""

    prompt = prompt_template.format(query=query, context=context)

    if model_name == "gemini":
        return call_gemini(prompt)
    elif model_name == "llama":
        return call_llama(prompt)
    elif model_name == "mistral":
        return call_mistral(prompt)
    elif model_name == "cohere":
        return call_cohere(prompt)
    else:
        raise ValueError("Unsupported model: " + model_name)

prompt_templates = {
    "default": """You are a knowledgeable financial news assistant with expertise in analyzing company announcements, stock trends, and macroeconomic events.

User Query:
{query}

Recent News Context:
{context}

Provide a concise, helpful answer based on the above news. Avoid hallucinating facts or sources.
""",

    "bullet_summary": """Summarize the key takeaways from the following financial news in bullet points relevant to this question:

Question: {query}

News Articles:
{context}
""",

    "analyst_voice": """Act like a professional stock analyst. Based on the news provided below, answer the question with precision and clarity.

Query: {query}

News Context:
{context}
"""
}

def compare_all_models_prompts(query, prompt_templates_dict):
    results = retrieve_with_decomposition(query, n_results=5)
    context = "\n\n".join([res["document"] for res in results])
    links = [res["metadata"].get("link") for res in results if res["metadata"].get("link")]

    models = ["gemini", "cohere", "llama", "mistral"]
    comparison_results = []

    for prompt_name, template in prompt_templates_dict.items():
        for model in models:
            print(f"Running → Model: {model}, Prompt: {prompt_name}")
            try:
                response = generate_rag_response_model_comp(query, context, model_name=model, prompt_template=template)
                if links:
                    response += "\n\nSources:\n" + "\n".join(links)
            except Exception as e:
                response = f"[ERROR] {str(e)}"

            comparison_results.append({
                "model": model,
                "prompt": prompt_name,
                "response": response
            })

    return comparison_results

query = "What are the latest developments from Apple and Nvidia related to AI and their impact on stock prices?"
all_outputs = compare_all_models_prompts(query, prompt_templates)

for result in all_outputs:
    print(f"\n\nPrompt: {result['prompt'].upper()} | Model: {result['model'].upper()}")
    print("-" * 80)
    print(result['response'])

df = pd.DataFrame(all_outputs)
df.to_csv("model_comparison.csv", index=False)
df.head()

