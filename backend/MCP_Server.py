import asyncio
import requests
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool
import mcp.server.stdio
import mcp.types

import os
import httpx
from googleapiclient.discovery import build
import google.generativeai as genai

from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Environment variables and constants
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DEFAULT_CSE_ID = os.getenv("GOOGLE_CSE_ID")
DEFAULT_PORT = int(os.getenv("MY_PORT", 8003))
DEFAULT_SEARCH_COUNT = int(os.getenv("MY_DEFAULT_SEARCH_COUNT", 10))
# FastAPI app
app = FastAPI(
    title="FastAPI + FastMCP Web Search Server",
    description="A server providing only the web search functionality.",
    version="1.0.0"
)
# Data models for FastAPI
class StatusResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    services: Dict[str, str]

class SearchRequest(BaseModel):
    query: str
    count: Optional[int] = None
    cse_id: Optional[str] = None

class SearchResult(BaseModel):
    title: str
    snippet: str
    link: str

class SearchResponse(BaseModel):
    results: list[SearchResult]
# FastAPI Routes
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with basic information"""
    return {
        "message": "FastAPI Web Search Server",
        "swagger_ui": "/docs",
        "redoc": "/redoc",
        "available_endpoints": ["/mcp/Websearch"]
    }

@app.get("/status", response_model=StatusResponse, tags=["System"])
async def get_status():
    """Get server status and health information"""
    return StatusResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        services={
            "fastapi": "running",
            "swagger_ui": "available"
        }
    )

@app.post("/mcp/Websearch", response_model=SearchResponse, tags=["Web Search"])
async def mcp_websearch(request: SearchRequest):
    logging.getLogger('googleapiclient.discovery_cache').setLevel(logging.ERROR)
    cse_id = request.cse_id if request.cse_id else DEFAULT_CSE_ID
    num_count = request.count if request.count else DEFAULT_SEARCH_COUNT

    try:
        logger.info(f"Starting Google Search for: {request.query}")
        service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)

        response = service.cse().list(
            q=request.query,
            cx=cse_id,
            num=num_count
        ).execute()

        logger.info("Search complete. Returning results.")

        results = []
        for item in response.get("items", []):
            results.append({
                "title": item["title"],
                "snippet": item.get("snippet", ""),
                "link": item["link"],
            })

        final_results = [SearchResult(**r) for r in results]
        return SearchResponse(results=final_results)

    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail="Search failed")

# MCP Tools (No longer using mcp server, but keeping FastAPI for consistency)
# If you are, you would need to define just the 'search' tool.
# Server management
async def main():
    """Main function to run the server"""
    config = uvicorn.Config(
        app=app,
        host="0.0.0.0",
        port=DEFAULT_PORT,
        log_level="info"
    )
    server = uvicorn.Server(config)
    
    print(f"Starting FastAPI Web Search Server...")
    print(f"FastAPI will be available at: http://localhost:{DEFAULT_PORT}")
    print(f"Swagger UI at: http://localhost:{DEFAULT_PORT}/docs")
    print("Press CTRL+C to stop the server.")

    try:
        await server.serve()
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.error(f"Server error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("MCP_Server:app", host="0.0.0.0", port=8003, reload=True)