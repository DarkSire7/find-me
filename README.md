# DisasterIQ ML Backend

The DisasterIQ Machine Learning Backend is a high-performance FastAPI application designed to power the intelligent matching and AI capabilities of the DisasterIQ platform. It provides REST API endpoints for multi-factor string matching, AI-powered physical tag extraction, and facial recognition.

Designed for scalability and stateless execution, this microservice is optimized for deployment on Google Cloud Run.

## Features

- **Multi-Factor Matching Engine:** Ranks missing person candidates against found person reports using weighted composite scoring:
  - **Name similarity:** Jaro-Winkler distance via `rapidfuzz` (40% weight)
  - **Age similarity:** Gaussian decay penalty (20% weight)
  - **Location proximity:** Haversine distance for geospatial coordinate decay (25% weight)
  - **Physical Tag matching:** Jaccard similarity index (15% weight)
- **AI Physical Tagging:** Automatically extracts structured physical descriptors (clothing, hair, age range, distinguishing marks) from raw images using Google's Gemini 1.5/2.0 Flash models.
- **Facial Recognition (Optional):** Employs `insightface` (ArcFace) to extract and compare deep facial embeddings.

## Project Structure

```text
disasteriq-ml/
├── main.py              # Main FastAPI application and routing logic
├── requirements.txt     # Python dependencies
├── Dockerfile           # Multi-stage production Docker build
└── README.md            # This documentation
```

## API Endpoints

Once the server is running, interactive Swagger documentation is automatically generated at `/docs`.

### `GET /health`
Liveness probe for Google Cloud Run health checks.
- **Response:** `{"status": "ok", "version": "2.0.0"}`

### `POST /match/strings`
Compares a single "found person" record against a list of "missing person" candidates and returns them sorted by match confidence.
- **Payload:**
  ```json
  {
    "found_person": { "id": "F1", "name": "...", "age": 30, "locationCoords": {"lat": 12.3, "lng": 45.6}, "tags": ["red shirt"] },
    "missing_persons": [ ... ]
  }
  ```
- **Response:** Array of matches sorted by `composite_score` (descending).

### `POST /extract-tags`
*(Deprecated for browser-direct integration, but available server-side)*
Analyzes an image URL using Gemini AI and returns an array of physical descriptors.
- **Payload:** `{"image_url": "https://..."}`
- **Response:** `{"physical_tags": ["red shirt", "glasses", "short hair"]}`

### `POST /match/faces`
Extracts 512-dimensional face embeddings from two image URLs and computes their cosine similarity.
- **Payload:** `{"image_url_1": "...", "image_url_2": "..."}`
- **Response:** `{"similarity_score": 0.85, "match": true}`

## Environment Variables

| Variable | Description |
|---|---|
| `GEMINI_API_KEY` | Your Google Gemini API key for the `/extract-tags` endpoint. If not provided, the endpoint will gracefully return a 503 status. |

## Local Development

1. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the development server:**
   ```bash
   uvicorn main:app --reload --port 8080
   ```
   The API will be accessible at `http://localhost:8080`.

## Production Deployment (Google Cloud Run)

The service is fully containerized and designed for GCP.

1. **Submit the build to Cloud Build:**
   ```bash
   gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/disasteriq-ml
   ```

2. **Deploy to Cloud Run:**
   ```bash
   gcloud run deploy disasteriq-ml \
     --image gcr.io/YOUR_PROJECT_ID/disasteriq-ml \
     --platform managed \
     --region asia-south1 \
     --allow-unauthenticated \
     --memory 2Gi \
     --update-env-vars GEMINI_API_KEY="your-api-key"
   ```

*(Note: Memory is set to 2Gi to comfortably accommodate the `insightface` ONNX models during inference).*
