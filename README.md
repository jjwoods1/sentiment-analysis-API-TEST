# Sentiment Analysis Tool

Advanced sentiment analysis tool using DistilBERT, RoBERTa, and Llama 2 models for both overall and contextual sentiment analysis.

## Features

- **REST API**: FastAPI-based REST API for easy integration
- **Overall Sentiment Analysis**: Analyze general sentiment of text using DistilBERT or RoBERTa models
- **Contextual Sentiment Analysis**: Analyze sentiment about specific contexts using Llama 2 7B with rule-based detection
- **Pattern Management**: Configurable sentiment patterns via text files or REST API endpoints
- **Docker Support**: Easy deployment with Docker and docker-compose
- **Multiple Input Formats**: Supports JSON requests, file uploads, plain text and JSON transcript files
- **Interactive Documentation**: Auto-generated API docs at `/docs`

## Requirements

- Python 3.11+
- Docker and docker-compose (for containerized deployment)
- 8GB+ RAM recommended (for LLM model)
- ~7GB disk space for Llama 2 model

## Installation

### Local Installation

```bash
pip install -r requirements.txt
```

### Docker Installation

```bash
docker-compose build
```

## Quick Start

### Run the API (Recommended)

```bash
# Build and start the API
docker-compose up -d

# API is now running at http://localhost:8000
# Interactive docs at http://localhost:8000/docs
```

### Stop the API

```bash
docker-compose down
```

## API Usage

### API Endpoints

Once running, access the API at `http://localhost:8000`:

#### Health & Documentation
- **GET `/`** - Health check
- **GET `/health`** - Detailed health status
- **GET `/docs`** - Interactive API documentation (Swagger UI)

#### Sentiment Analysis
- **POST `/analyze/overall`** - Analyze overall sentiment (JSON)
- **POST `/analyze/contextual`** - Analyze contextual sentiment (JSON)
- **POST `/analyze/overall/file`** - Analyze overall sentiment (file upload)
- **POST `/analyze/contextual/file`** - Analyze contextual sentiment (file upload)

#### Pattern Management
- **GET `/patterns/positive`** - Get list of positive sentiment patterns
- **GET `/patterns/negative`** - Get list of negative sentiment patterns
- **POST `/patterns/positive`** - Add a new positive sentiment pattern
- **POST `/patterns/negative`** - Add a new negative sentiment pattern
- **DELETE `/patterns/positive/{pattern}`** - Delete a specific positive pattern
- **DELETE `/patterns/negative/{pattern}`** - Delete a specific negative pattern
- **PUT `/patterns/positive`** - Replace all positive patterns
- **PUT `/patterns/negative`** - Replace all negative patterns

### Example API Requests

**Overall Sentiment Analysis (JSON):**
```bash
curl -X POST "http://localhost:8000/analyze/overall" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "I love this product! It works great.",
    "model": "roberta"
  }'
```

**Contextual Sentiment Analysis (JSON):**
```bash
curl -X POST "http://localhost:8000/analyze/contextual" \
  -H "Content-Type: application/json" \
  -d '{
    "context": "IONOS",
    "transcript": {
      "text": "I hate IONOS. Not happy with it at all.",
      "segments": [
        {
          "id": 1,
          "start": 0.0,
          "end": 5.0,
          "text": "I hate IONOS. Not happy with it at all."
        }
      ]
    }
  }'
```

**Overall Sentiment Analysis (File Upload):**
```bash
curl -X POST "http://localhost:8000/analyze/overall/file" \
  -F "file=@./data/input.txt" \
  -F "model=roberta"
```

**Contextual Sentiment Analysis (File Upload):**
```bash
curl -X POST "http://localhost:8000/analyze/contextual/file" \
  -F "file=@./data/transcription-example.json" \
  -F "context=IONOS"
```

**Get Positive Patterns:**
```bash
curl -X GET "http://localhost:8000/patterns/positive"
```

**Add a Positive Pattern:**
```bash
curl -X POST "http://localhost:8000/patterns/positive" \
  -H "Content-Type: application/json" \
  -d '{"pattern": "brilliant"}'
```

**Delete a Pattern:**
```bash
curl -X DELETE "http://localhost:8000/patterns/positive/brilliant"
```

**Replace All Negative Patterns:**
```bash
curl -X PUT "http://localhost:8000/patterns/negative" \
  -H "Content-Type: application/json" \
  -d '{"patterns": ["terrible", "awful", "horrible", "hate"]}'
```

### Using the Interactive Docs

1. Start the API: `docker-compose up -d`
2. Open browser: http://localhost:8000/docs
3. Click on any endpoint to test it
4. Click "Try it out" and enter your data
5. Click "Execute" to see results

## CLI Usage (Legacy)

### Local CLI Usage

**Overall Sentiment Analysis:**
```bash
python sentiment_analysis_tool.py --type overall --model roberta --file path/to/text.txt
```

**Contextual Sentiment Analysis:**
```bash
python sentiment_analysis_tool.py --type contextual --file path/to/transcript.json --context "Vodafone"
```

**Save output to file:**
```bash
python sentiment_analysis_tool.py --type overall --file input.txt --output results.json
```

### Docker CLI Usage

**Run overall sentiment analysis:**
```bash
docker-compose run --rm --profile cli sentiment-analysis-cli --type overall --model roberta --file /data/input.txt
```

**Run contextual sentiment analysis:**
```bash
docker-compose run --rm --profile cli sentiment-analysis-cli --type contextual --file /data/transcript.json --context "IONOS"
```

**Or use the batch scripts:**
```bash
# Windows
run-overall.bat input.txt roberta
run-contextual.bat IONOS transcription-example.json
```

**Note**: Place your input files in the `./data` directory - they'll be accessible at `/data/` inside the container.

## Command Line Arguments

- `--type`, `-t`: Analysis type (`overall` or `contextual`)
- `--model`, `-m`: Model for overall analysis (`distilbert` or `roberta`, default: `roberta`)
- `--file`, `-f`: Path to input file
- `--context`, `-c`: Context word for contextual analysis (required for contextual mode)
- `--output`, `-o`: Path to save output JSON (optional, prints to stdout if not provided)

## Input Formats

### Overall Sentiment Analysis

Accepts plain text files:
```
This is sample text for sentiment analysis...
```

### Contextual Sentiment Analysis

Requires JSON transcript files matching the schema in `transcript-schema.json`:
```json
{
  "text": "Full transcript text...",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 5.2,
      "text": "Segment text here"
    }
  ]
}
```

## Output Format

**Overall Sentiment:**
```json
{
  "overall_sentiment": "positive",
  "positive": 15,
  "neutral": 3,
  "negative": 2
}
```

**Contextual Sentiment:**
```json
{
  "context": "IONOS",
  "overall_sentiment": "negative",
  "positive": 2,
  "neutral": 5,
  "negative": 8,
  "segments": [
    {
      "segment-id": 1,
      "start": 30.0,
      "end": 37.38,
      "text": "I hate IONOS. Not happy with it at all.",
      "sentiment": "negative"
    }
  ]
}
```

## Models

- **DistilBERT**: `distilbert/distilbert-base-uncased-finetuned-sst-2-english`
- **RoBERTa**: `cardiffnlp/twitter-roberta-base-sentiment-latest`
- **Llama 2 7B**: Quantized GGUF model (auto-downloaded on first use)

The Llama 2 model (~4GB) is downloaded automatically to `.models/llm/` on first run of contextual analysis.

## Pattern Management

The tool uses configurable sentiment patterns for rule-based detection before LLM analysis. You can manage patterns in two ways:

### 1. Edit Pattern Files Directly

Edit `negative_patterns.txt` or `positive_patterns.txt`:
```txt
# Negative sentiment patterns
# One pattern per line
# Lines starting with # are comments

hate
terrible
awful
disappointed
```

### 2. Use the API

Manage patterns programmatically via REST API:
- View patterns: `GET /patterns/positive` or `GET /patterns/negative`
- Add pattern: `POST /patterns/positive` with `{"pattern": "excellent"}`
- Delete pattern: `DELETE /patterns/positive/excellent`
- Replace all: `PUT /patterns/positive` with `{"patterns": ["great", "awesome"]}`

Changes take effect immediately - no restart required!

## Docker Volumes

- `./models`: Persists downloaded models (~7GB)
- `./data`: Mount point for input/output files

## License

This project uses models with the following licenses:
- DistilBERT and RoBERTa: Apache 2.0
- Llama 2: Meta Llama 2 License
