#!/usr/bin/env python3
"""
Sentiment Analysis Tool

This script provides advanced sentiment analysis capabilities using:
1. Overall text sentiment analysis with DistilBERT or RoBERTa
2. Contextual sentiment analysis with Llama 2 7B

For usage instructions, see the accompanying README.md file or run with --help.
"""

import os
import sys
import subprocess
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Union, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("sentiment_analysis")

# Constants
BASE_DIR = Path(__file__).parent.resolve()
LLM_DIR = BASE_DIR / ".models" / "llm"
LLM_MODEL_NAME = "llama-2-7b.Q4_K_M.gguf"
LLM_URL = "https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_K_M.gguf"
MAX_TOKENS_FAST_MODEL = 512
FAST_MODEL_NAME = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
ROBERTA_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
MAX_LLM_CONTEXT = 512
MAX_RESPONSE_TOKENS = 300
NEGATIVE_PATTERNS_FILE = BASE_DIR / "negative_patterns.txt"
POSITIVE_PATTERNS_FILE = BASE_DIR / "positive_patterns.txt"

# ---------------------------
# Imports - ML/AI dependencies
# ---------------------------
try:
    import torch
    from transformers import pipeline, AutoTokenizer
    from llama_cpp import Llama
    from tqdm import tqdm
    import requests
except ImportError as e:
    logger.error(f"Missing required dependency: {e}")
    logger.error("Please install dependencies using: pip install -r requirements.txt")
    sys.exit(1)

# ---------------------------
# Utility Functions
# ---------------------------
def download_llm_model() -> Path:
    """Download the LLM model if not already present."""
    LLM_DIR.mkdir(parents=True, exist_ok=True)
    model_path = LLM_DIR / LLM_MODEL_NAME
    
    if not model_path.exists():
        logger.info(f"Downloading {LLM_MODEL_NAME}...")
        response = requests.get(LLM_URL, stream=True)
        total = int(response.headers.get("content-length", 0))
        
        with open(model_path, "wb") as f, tqdm(
            desc=f"Downloading {LLM_MODEL_NAME}",
            total=total,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                bar.update(size)
                
    return model_path

def load_patterns_from_file(file_path: Path) -> List[str]:
    """
    Load sentiment patterns from a text file.

    Args:
        file_path: Path to the patterns file

    Returns:
        List of patterns (lines that don't start with # and aren't empty)
    """
    patterns = []
    try:
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    # Skip empty lines and comments
                    if line and not line.startswith('#'):
                        patterns.append(line.lower())
            logger.info(f"Loaded {len(patterns)} patterns from {file_path.name}")
        else:
            logger.warning(f"Pattern file not found: {file_path}")
    except Exception as e:
        logger.error(f"Error loading patterns from {file_path}: {e}")

    return patterns

def detect_sentiment_keywords(text: str, context: str) -> Optional[str]:
    """
    Use rule-based detection for strong sentiment indicators.

    Args:
        text: The text to analyze
        context: The context word to focus on

    Returns:
        "positive", "negative", or None if no strong indicators found
    """
    text_lower = text.lower()
    context_lower = context.lower()

    if context_lower in text_lower:
        # Load patterns from files
        negative_patterns = load_patterns_from_file(NEGATIVE_PATTERNS_FILE)
        positive_patterns = load_patterns_from_file(POSITIVE_PATTERNS_FILE)

        # Check for negative sentiment patterns
        for pattern in negative_patterns:
            if pattern in text_lower:
                logger.info(f"Rule-based detection found negative pattern '{pattern}' in text")
                return "negative"

        # Check for positive sentiment patterns
        for pattern in positive_patterns:
            if pattern in text_lower:
                logger.info(f"Rule-based detection found positive pattern '{pattern}' in text")
                return "positive"

    # No strong sentiment found
    return None

def validate_json_against_schema(json_data: Dict[str, Any], schema_file: str) -> bool:
    """Validate a JSON object against a schema."""
    try:
        with open(schema_file, 'r', encoding='utf-8') as f:
            schema = json.load(f)
        
        # Basic validation (more comprehensive validation would use jsonschema library)
        required_keys = schema.get("required", [])
        for key in required_keys:
            if key not in json_data:
                logger.error(f"Missing required key: {key}")
                return False
                
        # Check segments structure for transcripts
        if "segments" in json_data:
            for segment in json_data["segments"]:
                if "id" not in segment or "text" not in segment:
                    logger.error("Invalid segment structure: missing id or text")
                    return False
        
        return True
    except Exception as e:
        logger.error(f"Schema validation error: {e}")
        return False

def load_file(file_path: str) -> Dict[str, Any]:
    """Load and parse a file based on its extension."""
    try:
        with open(file_path, "r", encoding='utf-8') as f:
            if file_path.endswith(".json"):
                return json.load(f)
            else:
                # For plain text files, return a simple dict with text content
                return {"text": f.read()}
    except Exception as e:
        logger.error(f"Error loading file {file_path}: {e}")
        sys.exit(1)

# ---------------------------
# Sentiment Analysis Functions
# ---------------------------
def analyze_overall_sentiment(file_path: str, model_name: str = "roberta") -> Dict[str, Any]:
    """
    Analyze overall sentiment of text using the specified model.
    
    Args:
        file_path: Path to the file to analyze
        model_name: Model to use - either 'distilbert' or 'roberta'
    """
    logger.info(f"Starting overall sentiment analysis using {model_name} model...")
    
    # Load the file content
    data = load_file(file_path)
    text = data.get("text", "")
    
    if not text:
        logger.error(f"No text content found in {file_path}")
        sys.exit(1)
    
    # Select the model to use
    if model_name == "distilbert":
        model = FAST_MODEL_NAME
        tokenizer_name = FAST_MODEL_NAME
    else:  # default to roberta
        model = ROBERTA_MODEL_NAME
        tokenizer_name = ROBERTA_MODEL_NAME
    
    # Initialize tokenizer for chunking text
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    def chunk_text(text, chunk_size=MAX_TOKENS_FAST_MODEL):
        tokens = tokenizer.encode(text, truncation=False)
        for i in range(0, len(tokens), chunk_size):
            yield tokenizer.decode(tokens[i:i + chunk_size], clean_up_tokenization_spaces=True)
    
    # Set up sentiment analysis pipeline
    device = 0 if torch.cuda.is_available() else -1
    sentiment_pipe = pipeline(
        "sentiment-analysis",
        model=model,
        device=device,
        tokenizer=tokenizer_name
    )
    
    # Process text in chunks
    chunks = list(chunk_text(text))
    results = []
    
    for chunk in tqdm(chunks, desc="Analyzing sentiment", ncols=80):
        res = sentiment_pipe(chunk, truncation=True, max_length=MAX_TOKENS_FAST_MODEL)
        results.extend(res)
    
    # Count sentiment results based on model output format
    if model_name == "distilbert":
        pos = sum(1 for r in results if r['label'].lower().startswith("pos"))
        neg = sum(1 for r in results if r['label'].lower().startswith("neg"))
        neutral = sum(1 for r in results if r['label'].lower().startswith("neu"))
    else:  # roberta model uses different label format
        pos = sum(1 for r in results if r['label'].lower() == "positive")
        neg = sum(1 for r in results if r['label'].lower() == "negative")
        neutral = sum(1 for r in results if r['label'].lower() == "neutral")
    
    # Determine overall sentiment
    if pos > neg and pos > neutral:
        overall = "positive"
    elif neg > pos and neg > neutral:
        overall = "negative"
    else:
        overall = "neutral"
    
    # Format results according to specified output structure
    output = {
        "overall_sentiment": overall,
        "positive": pos,
        "neutral": neutral,
        "negative": neg,
    }
    
    logger.info(f"Overall sentiment analysis complete: {overall}")
    return output

def analyze_contextual_sentiment(file_path: str, context: str) -> Dict[str, Any]:
    """Analyze contextual sentiment using Llama 2 model combined with rule-based detection."""
    logger.info(f"Starting contextual sentiment analysis for context: {context}...")
    
    # Validate the input file against the transcript schema
    schema_path = os.path.join(BASE_DIR, "transcript-schema.json")
    data = load_file(file_path)
    
    if not validate_json_against_schema(data, schema_path):
        logger.error(f"Invalid transcript format in {file_path}")
        sys.exit(1)
    
    # Extract segments from the transcript
    segments = data.get("segments", [])
    if not segments:
        logger.error("No segments found in the transcript")
        sys.exit(1)
    
    # Find segments containing the context word
    context_segments = []
    for segment in segments:
        if context.lower() in segment.get("text", "").lower():
            context_segments.append(segment)
    
    if not context_segments:
        logger.warning(f"Context '{context}' not found in any segment")
        return {
            "context": context,
            "overall_sentiment": "neutral",
            "positive": 0,
            "neutral": 0,
            "negative": 0,
            "segments": []
        }
    
    # Download and load the LLM model
    logger.info("Loading Llama model for contextual analysis...")
    model_path = download_llm_model()
    
    # Suppress llama_cpp logs and prints
    logging.getLogger("llama_cpp").setLevel(logging.CRITICAL)
    
    class DevNull:
        def write(self, msg): pass
        def flush(self): pass
    
    # Load the LLM model
    llm = None
    try:
        _stdout, _stderr = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = DevNull()
        llm = Llama(model_path=str(model_path), n_ctx=1024)
    finally:
        sys.stdout, sys.stderr = _stdout, _stderr
    
    # Process each segment containing the context
    pos = neg = neutral = 0
    result_segments = []
    
    for segment in tqdm(context_segments, desc="Analyzing segments", ncols=80):
        segment_text = segment.get("text", "")
        
        # First check for strong sentiment indicators with rule-based approach
        rule_sentiment = detect_sentiment_keywords(segment_text, context)
        
        if rule_sentiment:
            # Strong rule-based sentiment found, use it
            sentiment = rule_sentiment
            logger.info(f"Using rule-based sentiment for segment {segment.get('id')}: {sentiment}")
        else:
            # No strong indicators, use enhanced LLM prompt
            prompt = f"""
You are an AI text analyst specialized in sentiment detection. You are given a text segment.
Your task is to analyze the sentiment expressed regarding a specific context.

Context: {context}

Text segment:
{segment_text}

Instructions:
- Pay CLOSE attention to strong emotional words like 'hate', 'love', 'awful', 'terrible', 'great'
- Look for explicit statements of satisfaction or dissatisfaction
- Focus specifically on sentiment about {context}
- Classify as one of: positive, negative, neutral
- If ANY negative sentiment is expressed about the context, classify as 'negative'
- If ANY positive sentiment is expressed about the context, classify as 'positive'
- Only classify as 'neutral' if there is truly NO sentiment expressed

Respond in this exact JSON format:
{{"sentiment": "<positive|negative|neutral>"}}
"""

            try:
                _stdout, _stderr = sys.stdout, sys.stderr
                sys.stdout = sys.stderr = DevNull()
                response = llm(prompt, max_tokens=MAX_RESPONSE_TOKENS)
            finally:
                sys.stdout, sys.stderr = _stdout, _stderr
            
            # Parse JSON response safely
            try:
                res_json = json.loads(response['choices'][0]['text'].strip())
                sentiment = res_json.get("sentiment", "neutral")
                logger.info(f"Using LLM-based sentiment for segment {segment.get('id')}: {sentiment}")
            except Exception:
                sentiment = "neutral"
        
        # Update counters
        if sentiment == "positive":
            pos += 1
        elif sentiment == "negative":
            neg += 1
        else:
            neutral += 1
        
        # Add segment to results with start and end times
        result_segments.append({
            "segment-id": segment.get("id"),
            "start": segment.get("start"),
            "end": segment.get("end"),
            "text": segment_text,
            "sentiment": sentiment
        })
    
    # Determine overall sentiment
    if pos > neg and pos > neutral:
        overall = "positive"
    elif neg > pos and neg > neutral:
        overall = "negative"
    else:
        overall = "neutral"
    
    # Format results according to specified output structure
    output = {
        "context": context,
        "overall_sentiment": overall,
        "positive": pos,
        "neutral": neutral,
        "negative": neg,
        "segments": result_segments
    }
    
    logger.info(f"Contextual sentiment analysis complete for '{context}': {overall}")
    return output

# ---------------------------
# Main CLI Interface
# ---------------------------
def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Advanced Sentiment Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --type overall --model roberta --file path/to/text.txt
  %(prog)s --type contextual --file path/to/transcript.json --context Vodafone
        """
    )
    
    parser.add_argument(
        "--type", "-t", 
        choices=["overall", "contextual"], 
        help="Type of sentiment analysis to perform"
    )
    
    parser.add_argument(
        "--model", "-m",
        choices=["distilbert", "roberta"],
        default="roberta",
        help="Model to use for overall sentiment analysis (default: roberta)"
    )
    
    parser.add_argument(
        "--file", "-f", 
        type=str, 
        help="Path to the input file"
    )
    
    parser.add_argument(
        "--context", "-c", 
        type=str, 
        help="Context word for contextual analysis (required for contextual analysis)"
    )
    
    parser.add_argument(
        "--output", "-o", 
        type=str, 
        help="Path to save the output JSON (if not provided, prints to stdout)"
    )
    
    args = parser.parse_args()
    
    # Interactive mode if required arguments are missing
    analysis_type = args.type or input("Enter analysis type (overall/contextual): ").strip().lower()
    
    if analysis_type not in ["overall", "contextual"]:
        logger.error("Invalid analysis type. Must be 'overall' or 'contextual'.")
        sys.exit(1)
    
    file_path = args.file or input("Enter path to file for analysis: ").strip()
    
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        sys.exit(1)
    
    # Run the appropriate analysis
    if analysis_type == "overall":
        model_name = args.model or input("Enter model name (distilbert/roberta): ").strip().lower() or "roberta"
        if model_name not in ["distilbert", "roberta"]:
            logger.error("Invalid model name. Must be 'distilbert' or 'roberta'.")
            sys.exit(1)
        
        result = analyze_overall_sentiment(file_path, model_name)
    else:  # contextual
        context = args.context or input("Enter context word for analysis: ").strip()
        if not context:
            logger.error("Context is required for contextual analysis.")
            sys.exit(1)
        
        result = analyze_contextual_sentiment(file_path, context)
    
    # Output the results
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        logger.info(f"Results saved to {args.output}")
    else:
        print(json.dumps(result, indent=2))

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unhandled error: {e}", exc_info=True)
        sys.exit(1)
