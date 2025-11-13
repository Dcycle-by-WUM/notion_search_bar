import os
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
import logging
import csv
from io import StringIO
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define models
EMBEDDING_MODEL = "text-embedding-3-large"  # For vector embeddings

# Load environment variables from .env file
load_dotenv()

# Initialize clients
openai_kwargs = {"api_key": os.getenv("OPENAI_API_KEY")}
openai_org = os.getenv("OPENAI_ORGANIZATION")
if openai_org:
    openai_kwargs["organization"] = openai_org

try:
    client = OpenAI(**openai_kwargs)
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index("notion-rag-large")
    logger.info("Successfully initialized clients and index")
except Exception as e:
    logger.error(f"Error initializing clients: {str(e)}")
    raise

def format_results_for_csv(results):
    """
    Format query results into a CSV-friendly format.
    
    Args:
        results (dict): Query results containing matches and their metadata
        
    Returns:
        str: CSV-formatted string
    """
    if not results['matches']:
        return ""
        
    # Create a StringIO object to write CSV data
    output = StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow(['Score', 'Submission Time', 'How Found', 'Opportunity'])
    
    # Write data rows
    for match in results['matches']:
        writer.writerow([
            f"{match['score']:.2f}",
            match['metadata'].get('submission_time', ''),
            match['metadata'].get('how_found', ''),
            match['metadata'].get('opportunity', '')
        ])
    
    return output.getvalue()

def query_notion_data(query_text, top_k=5):
    """
    Query the Notion data stored in Pinecone and return relevant results.
    
    Args:
        query_text (str): The query text to search for
        top_k (int): Number of results to return (default: 5)
        
    Returns:
        dict: Query results containing matches and their metadata
    """
    try:
        logger.info(f"Generating embedding for query: {query_text}")
        # Generate embedding for the query
        embedding_response = client.embeddings.create(
            input=query_text,
            model=EMBEDDING_MODEL
        )
        
        if not embedding_response or not embedding_response.data:
            logger.error("No embedding response received")
            return {'matches': []}
            
        query_embedding = embedding_response.data[0].embedding
        logger.info("Successfully generated embedding")
        
        # Query the index
        logger.info("Querying Pinecone index")
        res = index.query(
            vector=[query_embedding],
            top_k=top_k,
            include_metadata=True
        )
        
        if not res or 'matches' not in res:
            logger.error("No results returned from Pinecone")
            return {'matches': []}
            
        logger.info(f"Received {len(res['matches'])} matches from Pinecone")
        
        # Filter out empty results and convert to JSON-serializable format
        valid_matches = []
        for match in res['matches']:
            if match['metadata'].get('opportunity') or match['metadata'].get('how_found'):
                # Convert to a simple dictionary with only the data we need
                valid_match = {
                    'id': match.get('id', ''),
                    'score': float(match.get('score', 0.0)),
                    'metadata': {
                        'submission_time': match['metadata'].get('submission_time', ''),
                        'how_found': match['metadata'].get('how_found', ''),
                        'opportunity': match['metadata'].get('opportunity', '')
                    }
                }
                valid_matches.append(valid_match)
        
        logger.info(f"Filtered to {len(valid_matches)} valid matches")
        
        return {'matches': valid_matches}
    except Exception as e:
        logger.error(f"Error in query_notion_data: {str(e)}", exc_info=True)
        return {'matches': []}

def generate_final_analysis(query_text, matches):
    """
    Generate a final analysis from the top matches using an LLM.
    
    Args:
        query_text (str): The user's query to provide context to the model.
        matches (list): List of match dicts as returned by query_notion_data()['matches']
    
    Returns:
        str: The model-generated analysis text, or an empty string on failure.
    """
    try:
        if not matches:
            return ""
        # Build context block from matches
        context_blocks = []
        for m in matches:
            metadata = m.get('metadata', {})
            submission_time = metadata.get('submission_time', '')
            how_found = metadata.get('how_found', '')
            opportunity = metadata.get('opportunity', '')
            block = (
                f"Submission Time: {submission_time}\n"
                f"How Found: {how_found}\n"
                f"Opportunity: {opportunity}"
            )
            context_blocks.append(block)
        context = "\n\n".join(context_blocks)
        # Call chat completion to synthesize a final analysis
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant for Dcycle product teams. Dcycle is a B2B SaaS company that helps "
                        "companies manage their ESG data. You are given a list of opportunities and problems that have "
                        "been submitted by Dcycle customers. Your job is to analyze the opportunities and problems and "
                        "provide a concise, prioritized analysis that is most relevant to the user's query."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"User query: {query_text}\n\n"
                        f"Based on the following context, analyze the opportunities and problems and provide:\n"
                        f"- A brief summary of key themes\n- The top opportunities (bulleted)\n- Any notable gaps or risks\n\n"
                        f"Context:\n{context}"
                    )
                }
            ]
        )
        return response.choices[0].message.content if response and response.choices else ""
    except Exception as e:
        logger.error(f"Error generating final analysis: {str(e)}", exc_info=True)
        return ""

if __name__ == "__main__":
    # Example usage
    query = "Find the most relevant opportunities related to scope 3"
    results = query_notion_data(query)
    
    # Print results in console
    print(f"\nQuery: '{query}'")
    print("\nRelevant Results:")
    print("-" * 80)
    
    if not results['matches']:
        print("No results found.")
    else:
        for i, match in enumerate(results['matches'], 1):
            print(f"\nResult {i} (Score: {match['score']:.2f})")
            print("-" * 40)
            
            if match['metadata'].get('submission_time'):
                print(f"Submission Time: {match['metadata']['submission_time']}")
            if match['metadata'].get('how_found'):
                print(f"How Found: {match['metadata']['how_found']}")
            if match['metadata'].get('opportunity'):
                print(f"Opportunity: {match['metadata']['opportunity']}")
    
    print("\n" + "-" * 80)
    
    # Generate and save CSV
    csv_data = format_results_for_csv(results)
    if csv_data:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"query_results_{timestamp}.csv"
        with open(filename, 'w', newline='') as f:
            f.write(csv_data)
        print(f"\nResults saved to {filename}") 