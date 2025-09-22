#%pip install datasets tqdm pandas pinecone openai --quiet

import os
import time
from tqdm.auto import tqdm
from pandas import DataFrame
import random
import string
from notion_client import Client
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

# Import OpenAI client and initialize with your API key.
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Import Pinecone client and related specifications.
from pinecone import Pinecone
from pinecone import ServerlessSpec

print("Successfully imported all required packages.")

# Initialize Notion client
notion = Client(auth=os.getenv("NOTION_API_KEY"))

def get_notion_database_data(database_id):
    """Fetch data from a Notion database."""
    results = []
    has_more = True
    start_cursor = None
    
    while has_more:
        response = notion.databases.query(
            database_id=database_id,
            start_cursor=start_cursor,
            page_size=100
        )
        
        results.extend(response["results"])
        has_more = response["has_more"]
        start_cursor = response.get("next_cursor")
    
    return results

def process_notion_data(notion_data):
    """Process Notion data into a format suitable for embeddings."""
    processed_data = []
    
    for page in notion_data:
        # Extract the three specific columns with their correct names and types
        submission_time = page["properties"].get("Submission time", {}).get("created_time", "")
        
        # Get the rich_text content for "How did you find"
        how_found_rich_text = page["properties"].get("How did you find the opportunity or problem?", {}).get("rich_text", [])
        how_found = how_found_rich_text[0].get("text", {}).get("content", "") if how_found_rich_text else ""
        
        # Get the rich_text content for "What opportunity"
        opportunity_rich_text = page["properties"].get("What opportunity or problem did you find?", {}).get("rich_text", [])
        opportunity = opportunity_rich_text[0].get("text", {}).get("content", "") if opportunity_rich_text else ""
        
        processed_data.append({
            "submission_time": submission_time,
            "how_found": how_found,
            "opportunity": opportunity,
            "page_id": page["id"]
        })
    
    return processed_data

# Replace with your Notion database ID
DATABASE_ID = os.getenv("NOTION_DATABASE_ID")

# Fetch and process data from Notion
print("Fetching data from Notion...")
notion_data = get_notion_database_data(DATABASE_ID)
processed_data = process_notion_data(notion_data)
ds_dataframe = DataFrame(processed_data)

# Create merged text for embeddings
ds_dataframe['merged'] = ds_dataframe.apply(
    lambda row: f"Submission Time: {row['submission_time']}\nHow Found: {row['how_found']}\nOpportunity: {row['opportunity']}", axis=1
)

print(f"Retrieved {len(ds_dataframe)} pages from Notion")

MODEL = "text-embedding-3-large"  # Using OpenAI's large embedding model
# Compute an embedding for the first document to obtain the embedding dimension.
sample_embedding_resp = client.embeddings.create(
    input=[ds_dataframe['merged'].iloc[0]],
    model=MODEL
)
embed_dim = len(sample_embedding_resp.data[0].embedding)
print(f"Embedding dimension: {embed_dim}")

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Initialize Pinecone index
index_name = "notion-rag-large"  # New index name to avoid conflicts
if index_name in pc.list_indexes().names():
    print(f"Deleting existing index {index_name}...")
    pc.delete_index(index_name)

# Create a serverless spec for the index
spec = ServerlessSpec(
    cloud="aws",
    region="us-east-1"  # Using us-east-1 which is supported by the free plan
)
print(f"Creating new index {index_name} with dimension {embed_dim}...")
pc.create_index(
    name=index_name,
    dimension=embed_dim,
    metric="cosine",
    spec=spec
)

index = pc.Index(index_name)

# Get existing IDs from Pinecone
print("Fetching existing entries from Pinecone...")
existing_ids = set()

# List all vectors
vectors = index.list()
for vector in vectors:
    if isinstance(vector, dict) and 'id' in vector:
        existing_ids.add(vector['id'])

print(f"Found {len(existing_ids)} existing entries in Pinecone")

# Filter out entries that are already in Pinecone
new_entries = [entry for entry in processed_data if entry['page_id'] not in existing_ids]
ds_dataframe = DataFrame(new_entries)

if len(ds_dataframe) > 0:
    # Create merged text for embeddings
    ds_dataframe['merged'] = ds_dataframe.apply(
        lambda row: f"Submission Time: {row['submission_time']}\nHow Found: {row['how_found']}\nOpportunity: {row['opportunity']}", axis=1
    )

    print(f"Found {len(ds_dataframe)} new pages to process from Notion")

    # Generate and upload embeddings
    print("Generating and uploading embeddings...")
    for i, row in ds_dataframe.iterrows():
        # Generate embedding
        embedding = client.embeddings.create(
            input=[row['merged']],
            model=MODEL
        ).data[0].embedding
        
        # Upload to Pinecone
        index.upsert(
            vectors=[{
                "id": row['page_id'],
                "values": embedding,
                "metadata": {
                    "submission_time": row['submission_time'],
                    "how_found": row['how_found'],
                    "opportunity": row['opportunity']
                }
            }]
        )

    print("New embeddings uploaded successfully!")
else:
    print("No new entries to process.")

print(f"Index stats: {index.describe_index_stats()}")

def query_pinecone_index(client, index, model, query_text):
    # Generate an embedding for the query.
    query_embedding = client.embeddings.create(input=query_text, model=model).data[0].embedding

    # Query the index and return top 5 matches.
    res = index.query(vector=[query_embedding], top_k=5, include_metadata=True)
    print("Query Results:")
    for match in res['matches']:
        print(f"{match['score']:.2f}: Submission Time: {match['metadata'].get('submission_time', 'N/A')}")
        print(f"   How Found: {match['metadata'].get('how_found', 'N/A')}")
        print(f"   Opportunity: {match['metadata'].get('opportunity', 'N/A')}\n")
    return res

# Example query
query = "Find the most relevant opportunities related to data input"
query_pinecone_index(client, index, MODEL, query)

# Retrieve and concatenate top 3 match contexts.
matches = index.query(
    vector=[client.embeddings.create(input=query, model=MODEL).data[0].embedding],
    top_k=10,
    include_metadata=True
)['matches']

context = "\n\n".join(
    f"Submission Time: {m['metadata'].get('submission_time', '')}\n"
    f"How Found: {m['metadata'].get('how_found', '')}\n"
    f"Opportunity: {m['metadata'].get('opportunity', '')}"
    for m in matches
)

# Use the context to generate a final answer.
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant for Dcycle product teams. Dcycle is a B2B SaaS company that helps companies manage their ESG data. You are given a list of opportunities and problems that have been submitted by Dcycle customers. Your job is to analyze the opportunities and problems and provide a list of opportunities that are most relevant to the user's query."},
        {"role": "user", "content": f"Based on the following context, analyze the opportunities and problems:\n\n{context}"}
    ]
)
print("\nFinal Analysis:")
print(response.choices[0].message.content)