
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests, os, logging
from datetime import datetime
from dotenv import load_dotenv 
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# RAG setup (Ensure these environment variables are set or provided)
GENAI_BASE_URL = os.getenv("GENAI_BASE_URL")
COPILOT_ID = os.getenv("COPILOT_ID")
END_USER_ID = os.getenv("END_USER_ID")

# Credentials for JWT token fetch
CLIENT_ID = os.getenv("GENAI_CLIENT_ID")
CLIENT_SECRET = os.getenv("GENAI_CLIENT_SECRET")

def fetch_jwt_token():
    """
    Fetch JWT token from GenAI API using client_id and client_secret.
    """
    token_url = f"{GENAI_BASE_URL}/client/token"
    payload = {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET
    }
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json"
    }
    try:
        resp = requests.post(token_url, json=payload, headers=headers)
        resp.raise_for_status()
        token_data = resp.json()
        token = token_data.get("access_token")
        if not token:
            logger.error(f"Token response did not contain 'access_token': {token_data}")
            return None
        return token
    except Exception as e:
        logger.error(f"Failed to fetch JWT token: {e}")
        return None

# Fetch and set JWT_TOKEN at startup
JWT_TOKEN = fetch_jwt_token()

# Add this line to create the FastAPI app
app = FastAPI(title="GenAI RAG Server")
class SearchQuery(BaseModel):
    query: str
    end_user_id: str = END_USER_ID

@app.post("/searchrag", summary="Search using GenAI RAG", tags=["GenAI RAG"])
async def search_mcp_tool(search_data: SearchQuery):
    """
    Performs a search query using the GenAI Copilot service.
    Returns only the available structured fields from the response.
    """
    if not JWT_TOKEN or JWT_TOKEN == "PLACEHOLDER_INVALID_TOKEN":
        logger.error("Attempted /search with missing or placeholder JWT_TOKEN.")
        raise HTTPException(status_code=401, detail="Authentication token (JWT_TOKEN) is missing or invalid. Please configure it.")

    url = f"{GENAI_BASE_URL}/copilots/{COPILOT_ID}/preview"
    headers = {
        "Authorization": f"Bearer {JWT_TOKEN}"
    }
    payload = {
        "query": (None, search_data.query),
        "end_user_id": (None, search_data.end_user_id)
    }

    logger.info(f"Sending search to {url} with query: {search_data.query}")

    try:
        response = requests.post(url, headers=headers, files=payload)
        response.raise_for_status()

        full_response = response.json()
        logger.info(" Search response received")

        # Corrected keys to extract based on GenAI Copilot API response structure
        keys_to_include = {
            "output_text": "output_text",
            "relevance_scores": "output_relevance_scores",
            "source_files": "output_file_names", # GenAI calls these "output_file_names"
            "grounded_qa": "output_groundings", # GenAI calls these "output_groundings"
            "suggested_followups": "output_next_questions" # GenAI calls these "output_next_questions"
        }

        safe_output = {}
        for friendly_key, actual_genai_key in keys_to_include.items():
            if actual_genai_key in full_response:
                safe_output[friendly_key] = full_response[actual_genai_key]
            # Optionally, you could also check within the first execution if the top-level is empty
            elif "executions" in full_response and full_response["executions"]:
                first_execution = full_response["executions"][0]
                if actual_genai_key in first_execution:
                    # Note: "output_file_names" vs "output_files" might be a subtle difference.
                    # Stick to the names seen at the top level for consistency if available.
                    safe_output[friendly_key] = first_execution[actual_genai_key]


        # Fallback: return everything if none of the *expected* structured keys were found,
        # or if you prefer to see the full response when debugging
        if not safe_output or len(safe_output) == 1 and "output_text" in safe_output:
            logger.warning("Only output_text or no expected structured keys found. Returning raw response for inspection.")
            return full_response

        return safe_output

    except requests.exceptions.HTTPError as e:
        error_detail = f"GenAI API Error: {e.response.status_code} - {e.response.text}"
        logger.error(f"HTTPError: {error_detail}")
        raise HTTPException(status_code=e.response.status_code, detail=error_detail)

    except requests.exceptions.ConnectionError as e:
        logger.error(f"ConnectionError: {e}")
        raise HTTPException(status_code=503, detail=f"Failed to connect to GenAI Copilot search service: {e}")
    except requests.exceptions.Timeout as e:
        logger.error(f"Timeout: {e}")
        raise HTTPException(status_code=504, detail=f"GenAI Copilot search service timed out: {e}")
    except requests.exceptions.RequestException as e:
        logger.error(f"RequestException: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected request error occurred during search: {e}")
    except Exception as e:
        logger.exception("Unexpected error during search:")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("Rag_server:app", host="0.0.0.0", port=8001, reload=True)