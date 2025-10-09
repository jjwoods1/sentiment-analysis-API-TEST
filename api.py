#!/usr/bin/env python3
"""
Sentiment Analysis API

FastAPI-based REST API for sentiment analysis services.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Import sentiment analysis functions
from sentiment_analysis_tool import (
    analyze_overall_sentiment,
    analyze_contextual_sentiment,
    validate_json_against_schema,
    load_patterns_from_file,
    BASE_DIR,
    NEGATIVE_PATTERNS_FILE,
    POSITIVE_PATTERNS_FILE
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("sentiment_api")

# Pydantic models for request/response
class OverallSentimentRequest(BaseModel):
    text: str = Field(..., description="Text content to analyze")
    model: str = Field("roberta", description="Model to use: distilbert or roberta")

class OverallSentimentResponse(BaseModel):
    overall_sentiment: str
    positive: int
    neutral: int
    negative: int

class TranscriptSegment(BaseModel):
    id: int
    start: float
    end: float
    text: str

class TranscriptData(BaseModel):
    text: str
    segments: List[TranscriptSegment]
    metadata: Optional[Dict[str, Any]] = None

class ContextualSentimentRequest(BaseModel):
    transcript: TranscriptData
    context: str = Field(..., description="Context word to analyze sentiment for")

class SegmentResult(BaseModel):
    segment_id: int = Field(alias="segment-id")
    start: float
    end: float
    text: str
    sentiment: str

    class Config:
        populate_by_name = True

class ContextualSentimentResponse(BaseModel):
    context: str
    overall_sentiment: str
    positive: int
    neutral: int
    negative: int
    segments: List[SegmentResult]

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str

class PatternListResponse(BaseModel):
    patterns: List[str]
    count: int

class PatternRequest(BaseModel):
    pattern: str = Field(..., description="Sentiment pattern to add", min_length=1)

class PatternListRequest(BaseModel):
    patterns: List[str] = Field(..., description="List of sentiment patterns")

# Global variable for models (loaded at startup)
models_loaded = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle events for the API"""
    # Startup
    logger.info("Starting Sentiment Analysis API...")
    logger.info("Models will be loaded on first request")
    yield
    # Shutdown
    logger.info("Shutting down Sentiment Analysis API...")

# Initialize FastAPI app
app = FastAPI(
    title="Sentiment Analysis API",
    description="Advanced sentiment analysis using DistilBERT, RoBERTa, and Llama 2 models",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - API health check"""
    return {
        "status": "healthy",
        "service": "Sentiment Analysis API",
        "version": "1.0.0"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Sentiment Analysis API",
        "version": "1.0.0"
    }

@app.post("/analyze/overall", response_model=OverallSentimentResponse)
async def analyze_overall(request: OverallSentimentRequest):
    """
    Analyze overall sentiment of text using DistilBERT or RoBERTa.

    - **text**: The text content to analyze
    - **model**: Model to use (distilbert or roberta, default: roberta)
    """
    try:
        logger.info(f"Overall sentiment analysis request with model: {request.model}")

        # Validate model choice
        if request.model not in ["distilbert", "roberta"]:
            raise HTTPException(
                status_code=400,
                detail="Invalid model. Must be 'distilbert' or 'roberta'"
            )

        # Create temporary file for analysis
        temp_file = Path("/tmp/temp_analysis.txt")
        temp_file.write_text(request.text, encoding='utf-8')

        # Run analysis
        result = analyze_overall_sentiment(str(temp_file), request.model)

        # Clean up
        temp_file.unlink()

        return result

    except Exception as e:
        logger.error(f"Error in overall sentiment analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/contextual", response_model=ContextualSentimentResponse)
async def analyze_contextual(request: ContextualSentimentRequest | List[ContextualSentimentRequest]):
    """
    Analyze contextual sentiment using Llama 2 model.

    - **transcript**: Transcript data with segments (or array containing single transcript)
    - **context**: Context word to analyze sentiment for
    """
    try:
        # Handle array-wrapped format: [{ context, transcript }] -> { context, transcript }
        if isinstance(request, list):
            if len(request) == 0:
                raise HTTPException(
                    status_code=400,
                    detail="Empty array in request"
                )
            actual_request = request[0]
        else:
            actual_request = request

        logger.info(f"Contextual sentiment analysis request for context: {actual_request.context}")

        # Validate context
        if not actual_request.context or not actual_request.context.strip():
            raise HTTPException(
                status_code=400,
                detail="Context word is required"
            )

        # Convert Pydantic model to dict format expected by analysis function
        import json
        transcript_dict = {
            "text": actual_request.transcript.text,
            "segments": [
                {
                    "id": seg.id,
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text
                }
                for seg in actual_request.transcript.segments
            ]
        }

        if actual_request.transcript.metadata:
            transcript_dict["metadata"] = actual_request.transcript.metadata

        # Validate against schema
        schema_path = os.path.join(BASE_DIR, "transcript-schema.json")
        if not validate_json_against_schema(transcript_dict, schema_path):
            raise HTTPException(
                status_code=400,
                detail="Invalid transcript format"
            )

        # Create temporary file for analysis
        temp_file = Path("/tmp/temp_transcript.json")
        temp_file.write_text(json.dumps(transcript_dict), encoding='utf-8')

        # Run analysis
        result = analyze_contextual_sentiment(str(temp_file), actual_request.context)

        # Clean up
        temp_file.unlink()

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in contextual sentiment analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/overall/file")
async def analyze_overall_file(
    file: UploadFile = File(...),
    model: str = Form("roberta")
):
    """
    Analyze overall sentiment from uploaded text file.

    - **file**: Text file to analyze
    - **model**: Model to use (distilbert or roberta, default: roberta)
    """
    try:
        logger.info(f"Overall sentiment analysis from file: {file.filename}")

        # Validate model choice
        if model not in ["distilbert", "roberta"]:
            raise HTTPException(
                status_code=400,
                detail="Invalid model. Must be 'distilbert' or 'roberta'"
            )

        # Read file content
        content = await file.read()
        text = content.decode('utf-8')

        # Create temporary file for analysis
        temp_file = Path("/tmp/temp_upload.txt")
        temp_file.write_text(text, encoding='utf-8')

        # Run analysis
        result = analyze_overall_sentiment(str(temp_file), model)

        # Clean up
        temp_file.unlink()

        return result

    except Exception as e:
        logger.error(f"Error in file upload analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/contextual/file")
async def analyze_contextual_file(
    file: UploadFile = File(...),
    context: str = Form(...)
):
    """
    Analyze contextual sentiment from uploaded JSON transcript file.

    - **file**: JSON transcript file to analyze
    - **context**: Context word to analyze sentiment for
    """
    try:
        logger.info(f"Contextual sentiment analysis from file: {file.filename}")

        # Validate context
        if not context or not context.strip():
            raise HTTPException(
                status_code=400,
                detail="Context word is required"
            )

        # Read and validate file content
        content = await file.read()
        logger.info(f"Received file size: {len(content)} bytes")

        import json
        try:
            data = json.loads(content.decode('utf-8'))
            logger.info(f"Parsed JSON type: {type(data)}")

            if isinstance(data, list):
                logger.info(f"Data is array with {len(data)} elements")
            else:
                logger.info(f"Data is object with keys: {list(data.keys()) if isinstance(data, dict) else 'not a dict'}")

            # Handle array-wrapped format: [{ transcript }] -> { transcript }
            if isinstance(data, list):
                if len(data) == 0:
                    raise HTTPException(
                        status_code=400,
                        detail="Empty array in JSON file"
                    )
                transcript_dict = data[0]
                logger.info(f"Unwrapped array, transcript_dict keys: {list(transcript_dict.keys())}")
            else:
                transcript_dict = data
                logger.info(f"Using data directly, keys: {list(transcript_dict.keys())}")

        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid JSON file: {str(e)}"
            )

        # Log the transcript structure for debugging
        logger.info(f"Final transcript_dict keys: {list(transcript_dict.keys())}")
        logger.info(f"Has 'text' key: {'text' in transcript_dict}")
        logger.info(f"Has 'segments' key: {'segments' in transcript_dict}")
        if 'segments' in transcript_dict:
            logger.info(f"Transcript has {len(transcript_dict.get('segments', []))} segments")

        # Validate against schema
        schema_path = os.path.join(BASE_DIR, "transcript-schema.json")
        if not validate_json_against_schema(transcript_dict, schema_path):
            logger.error(f"Schema validation failed for transcript with keys: {list(transcript_dict.keys())}")
            raise HTTPException(
                status_code=400,
                detail="Invalid transcript format - check server logs for details"
            )

        # Create temporary file for analysis
        temp_file = Path("/tmp/temp_upload.json")
        temp_file.write_text(json.dumps(transcript_dict), encoding='utf-8')

        # Run analysis
        result = analyze_contextual_sentiment(str(temp_file), context)

        # Clean up
        temp_file.unlink()

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in file upload analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------
# Pattern Management Endpoints
# ---------------------------

@app.get("/patterns/positive", response_model=PatternListResponse)
async def get_positive_patterns():
    """Get list of positive sentiment patterns"""
    try:
        patterns = load_patterns_from_file(POSITIVE_PATTERNS_FILE)
        return {
            "patterns": patterns,
            "count": len(patterns)
        }
    except Exception as e:
        logger.error(f"Error reading positive patterns: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/patterns/negative", response_model=PatternListResponse)
async def get_negative_patterns():
    """Get list of negative sentiment patterns"""
    try:
        patterns = load_patterns_from_file(NEGATIVE_PATTERNS_FILE)
        return {
            "patterns": patterns,
            "count": len(patterns)
        }
    except Exception as e:
        logger.error(f"Error reading negative patterns: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/patterns/positive", response_model=PatternListResponse)
async def add_positive_pattern(request: PatternRequest):
    """
    Add a new positive sentiment pattern

    - **pattern**: The pattern to add
    """
    try:
        pattern = request.pattern.strip().lower()

        # Load existing patterns
        patterns = load_patterns_from_file(POSITIVE_PATTERNS_FILE)

        # Check if pattern already exists
        if pattern in patterns:
            raise HTTPException(status_code=400, detail="Pattern already exists")

        # Add new pattern to file
        with open(POSITIVE_PATTERNS_FILE, 'a', encoding='utf-8') as f:
            f.write(f"\n{pattern}")

        logger.info(f"Added positive pattern: {pattern}")

        # Return updated list
        patterns = load_patterns_from_file(POSITIVE_PATTERNS_FILE)
        return {
            "patterns": patterns,
            "count": len(patterns)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding positive pattern: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/patterns/negative", response_model=PatternListResponse)
async def add_negative_pattern(request: PatternRequest):
    """
    Add a new negative sentiment pattern

    - **pattern**: The pattern to add
    """
    try:
        pattern = request.pattern.strip().lower()

        # Load existing patterns
        patterns = load_patterns_from_file(NEGATIVE_PATTERNS_FILE)

        # Check if pattern already exists
        if pattern in patterns:
            raise HTTPException(status_code=400, detail="Pattern already exists")

        # Add new pattern to file
        with open(NEGATIVE_PATTERNS_FILE, 'a', encoding='utf-8') as f:
            f.write(f"\n{pattern}")

        logger.info(f"Added negative pattern: {pattern}")

        # Return updated list
        patterns = load_patterns_from_file(NEGATIVE_PATTERNS_FILE)
        return {
            "patterns": patterns,
            "count": len(patterns)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding negative pattern: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/patterns/positive/{pattern}")
async def delete_positive_pattern(pattern: str):
    """
    Delete a positive sentiment pattern

    - **pattern**: The pattern to delete
    """
    try:
        pattern_lower = pattern.strip().lower()

        # Load existing patterns
        patterns = load_patterns_from_file(POSITIVE_PATTERNS_FILE)

        # Check if pattern exists
        if pattern_lower not in patterns:
            raise HTTPException(status_code=404, detail="Pattern not found")

        # Remove the pattern
        patterns.remove(pattern_lower)

        # Rewrite file
        with open(POSITIVE_PATTERNS_FILE, 'w', encoding='utf-8') as f:
            f.write("# Positive sentiment patterns\n")
            f.write("# One pattern per line\n")
            f.write("# Lines starting with # are comments and will be ignored\n\n")
            for p in patterns:
                f.write(f"{p}\n")

        logger.info(f"Deleted positive pattern: {pattern_lower}")

        return {
            "message": "Pattern deleted successfully",
            "pattern": pattern_lower,
            "remaining_count": len(patterns)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting positive pattern: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/patterns/negative/{pattern}")
async def delete_negative_pattern(pattern: str):
    """
    Delete a negative sentiment pattern

    - **pattern**: The pattern to delete
    """
    try:
        pattern_lower = pattern.strip().lower()

        # Load existing patterns
        patterns = load_patterns_from_file(NEGATIVE_PATTERNS_FILE)

        # Check if pattern exists
        if pattern_lower not in patterns:
            raise HTTPException(status_code=404, detail="Pattern not found")

        # Remove the pattern
        patterns.remove(pattern_lower)

        # Rewrite file
        with open(NEGATIVE_PATTERNS_FILE, 'w', encoding='utf-8') as f:
            f.write("# Negative sentiment patterns\n")
            f.write("# One pattern per line\n")
            f.write("# Lines starting with # are comments and will be ignored\n\n")
            for p in patterns:
                f.write(f"{p}\n")

        logger.info(f"Deleted negative pattern: {pattern_lower}")

        return {
            "message": "Pattern deleted successfully",
            "pattern": pattern_lower,
            "remaining_count": len(patterns)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting negative pattern: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/patterns/positive", response_model=PatternListResponse)
async def replace_positive_patterns(request: PatternListRequest):
    """
    Replace all positive sentiment patterns

    - **patterns**: List of patterns to set
    """
    try:
        # Validate and clean patterns
        patterns = [p.strip().lower() for p in request.patterns if p.strip()]

        # Remove duplicates while preserving order
        seen = set()
        unique_patterns = []
        for p in patterns:
            if p not in seen:
                seen.add(p)
                unique_patterns.append(p)

        # Write to file
        with open(POSITIVE_PATTERNS_FILE, 'w', encoding='utf-8') as f:
            f.write("# Positive sentiment patterns\n")
            f.write("# One pattern per line\n")
            f.write("# Lines starting with # are comments and will be ignored\n\n")
            for p in unique_patterns:
                f.write(f"{p}\n")

        logger.info(f"Replaced positive patterns with {len(unique_patterns)} new patterns")

        return {
            "patterns": unique_patterns,
            "count": len(unique_patterns)
        }
    except Exception as e:
        logger.error(f"Error replacing positive patterns: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/patterns/negative", response_model=PatternListResponse)
async def replace_negative_patterns(request: PatternListRequest):
    """
    Replace all negative sentiment patterns

    - **patterns**: List of patterns to set
    """
    try:
        # Validate and clean patterns
        patterns = [p.strip().lower() for p in request.patterns if p.strip()]

        # Remove duplicates while preserving order
        seen = set()
        unique_patterns = []
        for p in patterns:
            if p not in seen:
                seen.add(p)
                unique_patterns.append(p)

        # Write to file
        with open(NEGATIVE_PATTERNS_FILE, 'w', encoding='utf-8') as f:
            f.write("# Negative sentiment patterns\n")
            f.write("# One pattern per line\n")
            f.write("# Lines starting with # are comments and will be ignored\n\n")
            for p in unique_patterns:
                f.write(f"{p}\n")

        logger.info(f"Replaced negative patterns with {len(unique_patterns)} new patterns")

        return {
            "patterns": unique_patterns,
            "count": len(unique_patterns)
        }
    except Exception as e:
        logger.error(f"Error replacing negative patterns: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Run the API server
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")

    logger.info(f"Starting API server on {host}:{port}")
    uvicorn.run(
        "api:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )
