"""
DisasterIQ ML Backend — Matching Engine
Google Solution Challenge
FastAPI application for Cloud Run deployment.

Endpoints:
  POST /match/strings  — Weighted string + geo + tag similarity ranking
  POST /match/faces    — Face encoding comparison via insightface/face_recognition
"""

from __future__ import annotations

import logging
import math
import os
import urllib.request
from io import BytesIO
from typing import Any, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, status, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from rapidfuzz.distance import JaroWinkler
import google.generativeai as genai
import json
import PIL.Image

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger("disasteriq")

# ---------------------------------------------------------------------------
# App bootstrap
# ---------------------------------------------------------------------------
app = FastAPI(
    title="DisasterIQ ML Engine",
    description="High-performance matching engine for missing/found person reports.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gemini AI setup
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
else:
    logger.warning("GEMINI_API_KEY not found in environment variables. AI tagging will fail.")
    gemini_model = None

# ---------------------------------------------------------------------------
# Global Error Handling & CORS Enforcement
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Ensures even hard crashes return JSON + CORS headers."""
    logger.error(f"Unhandled Exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "INTERNAL_SERVER_ERROR",
            "message": "The server encountered an error processing your request.",
            "detail": str(exc) if os.getenv("DEBUG") == "true" else "Check server logs."
        },
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "*",
            "Access-Control-Allow-Headers": "*"
        }
    )

@app.middleware("http")
async def force_cors_middleware(request: Request, call_next):
    """Supplementary middleware to ensure headers are present on every response."""
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response

# ---------------------------------------------------------------------------
# Face recognition backend (optional — graceful fallback if unavailable)
# ---------------------------------------------------------------------------
try:
    import cv2
    from insightface.app import FaceAnalysis
    FACE_BACKEND = "insightface"
    logger.info("Face backend: insightface (ArcFace)")
    
    # Initialize the FaceAnalysis app (buffalo_l is the highly accurate default model)
    # providers=['CPUExecutionProvider'] avoids warnings if ONNX Runtime doesn't have GPU
    face_app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
    face_app.prepare(ctx_id=-1, det_size=(640, 640))
except ImportError as e:
    face_app = None
    FACE_BACKEND = "none"
    logger.warning(f"insightface not available ({e}) — face endpoint will return 503.")

# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

class GeoPoint(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)


class PersonRecord(BaseModel):
    """Shared schema for both Missing and Found person records."""
    id: str
    name: Optional[str] = None
    age: Optional[int] = Field(default=None, ge=0, le=150)
    gender: Optional[str] = None
    location: Optional[GeoPoint] = None
    # Physical descriptors: ["tall", "short hair", "tattoo on left arm", …]
    physical_tags: list[str] = Field(default_factory=list)
    # URL of a stored photo (Firebase Storage)
    photo_url: Optional[str] = None


class StringMatchRequest(BaseModel):
    found_person: PersonRecord
    missing_persons: list[PersonRecord] = Field(..., min_length=1)


class MatchResult(BaseModel):
    missing_person_id: str
    missing_person_name: Optional[str]
    composite_score: float = Field(..., ge=0.0, le=1.0)
    name_score: float
    age_score: float
    location_score: float
    tag_score: float
    estimated_distance_km: Optional[float]


class StringMatchResponse(BaseModel):
    found_person_id: str
    matches: list[MatchResult]


class FaceMatchRequest(BaseModel):
    image_url_1: str
    image_url_2: str

    @field_validator("image_url_1", "image_url_2")
    @classmethod
    def must_be_http(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError("Image URLs must start with http:// or https://")
        return v


class FaceMatchResponse(BaseModel):
    is_match: bool
    similarity_percentage: float = Field(..., ge=0.0, le=100.0)
    face_distance: Optional[float]
    backend: str


class TagExtractionRequest(BaseModel):
    image_url: str


class TagExtractionResponse(BaseModel):
    physical_tags: list[str] = Field(default_factory=list)
    error: Optional[str] = None
    warning: Optional[str] = None


# ---------------------------------------------------------------------------
# Core Algorithms
# ---------------------------------------------------------------------------

# Composite score weights (must sum to 1.0)
WEIGHTS = {
    "name":     0.40,
    "age":      0.20,
    "location": 0.25,
    "tags":     0.15,
}

# Threshold above which a face comparison is considered a match (Cosine Similarity)
# ArcFace (buffalo_l) threshold is typically ~0.35 - 0.45 for cosine similarity in wild conditions.
FACE_MATCH_THRESHOLD = 0.35


def jaro_winkler_score(a: Optional[str], b: Optional[str]) -> float:
    """Normalised Jaro-Winkler similarity [0, 1]. Returns 0 if either is None."""
    if not a or not b:
        return 0.0
    # rapidfuzz returns a value in [0, 1]
    return JaroWinkler.normalized_similarity(a.strip().lower(), b.strip().lower())


def haversine_km(p1: GeoPoint, p2: GeoPoint) -> float:
    """
    Great-circle distance between two GeoPoints (Haversine formula).
    Returns distance in kilometres.
    """
    R = 6_371.0  # Earth radius in km
    lat1, lon1 = math.radians(p1.latitude), math.radians(p1.longitude)
    lat2, lon2 = math.radians(p2.latitude), math.radians(p2.longitude)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def location_score(found: Optional[GeoPoint], missing: Optional[GeoPoint]) -> tuple[float, Optional[float]]:
    """
    Converts distance to a proximity score in [0, 1].
    Uses an exponential decay: score = exp(-d / SIGMA).
    SIGMA = 50 km → at 50 km score ≈ 0.37, at 10 km ≈ 0.82, at 0 km = 1.0
    Returns (score, distance_km).
    """
    if found is None or missing is None:
        return 0.5, None  # neutral — no location data available
    SIGMA = 50.0
    dist = haversine_km(found, missing)
    score = math.exp(-dist / SIGMA)
    return score, dist


def age_score(found_age: Optional[int], missing_age: Optional[int]) -> float:
    """
    Gaussian proximity score based on age difference.
    Returns 0.5 (neutral) if either age is missing.
    Score decays as abs difference grows; at ±5 years ≈ 0.78, at ±10 ≈ 0.37.
    """
    if found_age is None or missing_age is None:
        return 0.5  # neutral
    diff = abs(found_age - missing_age)
    SIGMA = 7.0
    return math.exp(-(diff**2) / (2 * SIGMA**2))


def tag_jaccard_score(found_tags: list[str], missing_tags: list[str]) -> float:
    """
    Jaccard similarity between two sets of normalised physical descriptor tags.
    Returns 0.5 if both lists are empty (neutral), 0.0 if only one is empty.
    """
    a = {t.strip().lower() for t in found_tags}
    b = {t.strip().lower() for t in missing_tags}
    if not a and not b:
        return 0.5  # neutral — no tags provided
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union


def compute_composite_score(
    found: PersonRecord,
    missing: PersonRecord,
) -> MatchResult:
    n_score = jaro_winkler_score(found.name, missing.name)
    a_score = age_score(found.age, missing.age)
    l_score, dist_km = location_score(found.location, missing.location)
    t_score = tag_jaccard_score(found.physical_tags, missing.physical_tags)

    composite = (
        WEIGHTS["name"]     * n_score
        + WEIGHTS["age"]    * a_score
        + WEIGHTS["location"] * l_score
        + WEIGHTS["tags"]   * t_score
    )

    return MatchResult(
        missing_person_id=missing.id,
        missing_person_name=missing.name,
        composite_score=round(composite, 4),
        name_score=round(n_score, 4),
        age_score=round(a_score, 4),
        location_score=round(l_score, 4),
        tag_score=round(t_score, 4),
        estimated_distance_km=round(dist_km, 2) if dist_km is not None else None,
    )


# ---------------------------------------------------------------------------
# Face helpers
# ---------------------------------------------------------------------------

def _download_image_bytes(url: str, timeout: int = 10) -> bytes:
    """Download image bytes from a public URL."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "DisasterIQ-ML/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to download image from '{url}': {exc}",
        )


def _encode_face_from_bytes(image_bytes: bytes, url: str) -> np.ndarray:
    """Return a 512-d face encoding using insightface ArcFace."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img_bgr is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Could not decode image from '{url}'",
        )

    faces = face_app.get(img_bgr)
    if not faces:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"No face detected in image: '{url}'",
        )
    return faces[0].embedding


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", tags=["Health"])
def health_check() -> dict[str, Any]:
    return {
        "service": "DisasterIQ ML Engine",
        "status": "healthy",
        "face_backend": FACE_BACKEND,
        "gemini_configured": gemini_model is not None,
        "weights": WEIGHTS,
    }


@app.get("/health", tags=["Health"])
def liveness() -> dict[str, str]:
    """Cloud Run liveness probe."""
    return {"status": "ok"}


@app.post(
    "/match/strings",
    response_model=StringMatchResponse,
    tags=["Matching"],
    summary="Rank missing persons against a found person by composite similarity.",
)
def match_strings(payload: StringMatchRequest) -> StringMatchResponse:
    """
    Compute a Composite Confidence Score for each missing person relative to
    the found person, then return results sorted by descending score.

    Score breakdown:
    - **Name**     — 40 %  (Jaro-Winkler)
    - **Age**      — 20 %  (Gaussian proximity)
    - **Location** — 25 %  (Haversine + exponential decay)
    - **Tags**     — 15 %  (Jaccard on physical descriptors)
    """
    logger.info(
        "match/strings — found=%s  candidates=%d",
        payload.found_person.id,
        len(payload.missing_persons),
    )
    try:
        results = [
            compute_composite_score(payload.found_person, mp)
            for mp in payload.missing_persons
        ]
        results.sort(key=lambda r: r.composite_score, reverse=True)
    except Exception as exc:
        logger.exception("Unhandled error in match_strings for found=%s: %s", payload.found_person.id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "SCORING_FAILED",
                "message": "An internal error occurred during scoring. Please retry.",
            },
        )

    return StringMatchResponse(
        found_person_id=payload.found_person.id,
        matches=results,
    )


@app.post(
    "/match/faces",
    response_model=FaceMatchResponse,
    tags=["Matching"],
    summary="Compare two face images and return a similarity score.",
)
def match_faces(payload: FaceMatchRequest) -> FaceMatchResponse:
    """
    Downloads both images, extracts the primary face encoding from each, and
    computes the Cosine similarity. A similarity >= 0.45 is considered a
    match (configurable via FACE_MATCH_THRESHOLD).
    """
    if face_app is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Face recognition backend is not installed in this deployment.",
        )

    logger.info("match/faces — url1=%s  url2=%s", payload.image_url_1, payload.image_url_2)

    bytes1 = _download_image_bytes(payload.image_url_1)
    bytes2 = _download_image_bytes(payload.image_url_2)

    enc1 = _encode_face_from_bytes(bytes1, payload.image_url_1)
    enc2 = _encode_face_from_bytes(bytes2, payload.image_url_2)

    # Cosine Similarity between 512-d encodings
    dot_product = np.dot(enc1, enc2)
    norm_sq = np.linalg.norm(enc1) * np.linalg.norm(enc2)
    cos_sim = float(dot_product / norm_sq) if norm_sq > 0 else 0.0
    
    is_match = cos_sim >= FACE_MATCH_THRESHOLD

    # For ArcFace, cosine similarity of the same person across conditions 
    # natively lands in 0.35–0.80+. Different people are usually < 0.10.
    # We map this to a human-readable confidence score where the threshold = 80%
    if cos_sim < 0:
        similarity = 0.0
    elif cos_sim < FACE_MATCH_THRESHOLD:
        similarity = (cos_sim / FACE_MATCH_THRESHOLD) * 80.0
    else:
        similarity = 80.0 + ((cos_sim - FACE_MATCH_THRESHOLD) / (1.0 - FACE_MATCH_THRESHOLD)) * 20.0

    return FaceMatchResponse(
        is_match=is_match,
        similarity_percentage=round(similarity, 2),
        face_distance=round(1.0 - cos_sim, 4), # For schema compatibility
        backend=FACE_BACKEND,
    )


@app.post(
    "/extract-tags",
    response_model=TagExtractionResponse,
    tags=["Matching"],
    summary="Automatically extract physical descriptors from a person's image using Gemini 1.5 Flash.",
)
async def extract_tags(payload: TagExtractionRequest) -> TagExtractionResponse:
    """
    Downloads the image, passes it to Gemini 1.5 Flash to identify physical
    characteristics like clothing, hair, and distinguishing features.
    """
    if gemini_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Gemini AI is not configured (missing API key)."
        )

    logger.info("extract/tags — url=%s", payload.image_url)

    try:
        # Reuse existing download utility
        image_bytes = _download_image_bytes(payload.image_url)
        img = PIL.Image.open(BytesIO(image_bytes))

        # Precise prompt
        prompt = (
            "Analyze this photo of a person. Extract their physical descriptors "
            "(e.g., hair color, clothing, glasses, facial hair, approximate age). "
            "Return a JSON object with a single key \"physical_tags\" containing a list of string descriptors. "
            "Return ONLY valid JSON with no markdown formatting or explanation."
        )

        # Generate response
        response = await gemini_model.generate_content_async([prompt, img])
        
        # Clean response text
        if not response or not response.text:
             return {"physical_tags": [], "warning": "AI returned an empty response."}

        text_response = response.text.strip()
        # Remove markdown code blocks if present
        if text_response.startswith("```"):
            text_response = text_response.split("```")[1]
            if text_response.startswith("json"):
                text_response = text_response[4:].strip()

        parsed = json.loads(text_response)
        return TagExtractionResponse(physical_tags=parsed.get("physical_tags", []))

    except Exception as exc:
        error_msg = str(exc)
        logger.error(f"Gemini tag extraction failed: {error_msg}")
        return TagExtractionResponse(physical_tags=[], error=error_msg)


# ---------------------------------------------------------------------------
# Entry-point (local dev only — Cloud Run uses uvicorn via CMD)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8080")),
        reload=False,
        log_level="info",
    )
