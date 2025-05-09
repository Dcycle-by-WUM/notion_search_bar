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
try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
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