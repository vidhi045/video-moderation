# app.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from transformers import pipeline
from PIL import Image
import io
import tempfile
import os
import cv2
import json
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorGridFSBucket
from datetime import datetime
import time
from dotenv import load_dotenv

# ---------------- FASTAPI & MONGO SETUP ----------------
app = FastAPI(title="Video Moderation Service")

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
client = AsyncIOMotorClient(MONGO_URI)
db = client["ai-moderation"]

# Separate GridFS bucket for videos
video_fs_bucket = AsyncIOMotorGridFSBucket(db, bucket_name="videos")

# Separate collection for video results
video_results_collection = db["video_results"]


# ---------------- CONFIGURATION ----------------
UNSAFE_THRESHOLD = 0.35
VIDEO_BLOCK_PERCENT = 30
FPS_INTERVAL = 1
MODEL_NAME = "openai/clip-vit-large-patch14"

# ---------------- LOAD MODEL ----------------
pipe = pipeline(
    "zero-shot-image-classification",
    model=MODEL_NAME,
    framework="pt"
)

# ---------------- LOAD LABELS ----------------
def load_labels():
    with open("labels.json", "r") as f:
        data = json.load(f)
    return data["WEAPON_LABELS"], data["UNSAFE_LABELS"], data["SAFE_LABELS"]

WEAPON_LABELS, UNSAFE_LABELS, SAFE_LABELS = load_labels()
ALL_LABELS = WEAPON_LABELS + UNSAFE_LABELS + SAFE_LABELS

# ---------------- UTIL FUNCTIONS ----------------
def extract_frames(video_path, fps_interval=FPS_INTERVAL):
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise HTTPException(status_code=400, detail="Invalid or corrupted video file")

    frames = []
    fps = vid.get(cv2.CAP_PROP_FPS) or 30
    interval = max(int(fps * fps_interval), 1)

    count = 0
    success, frame = vid.read()
    while success:
        if count % interval == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
        success, frame = vid.read()
        count += 1

    vid.release()
    return frames

def classify_image_full(image: Image.Image):
    results = pipe(image, candidate_labels=ALL_LABELS)
    unsafe_scores, safe_scores, critical_scores = [], [], []

    for r in results:
        label, score = r["label"], r["score"]
        if label in WEAPON_LABELS:
            critical_scores.append((label, score))
        elif label in UNSAFE_LABELS:
            unsafe_scores.append((label, score))
        else:
            safe_scores.append((label, score))

    return {
        "max_critical_score": max([s for _, s in critical_scores], default=0),
        "max_unsafe_score": max([s for _, s in unsafe_scores], default=0),
        "unsafe_labels": [l for l, _ in unsafe_scores],
        "safe_labels": [l for l, _ in safe_scores],
        "top_predictions": sorted(results, key=lambda x: x["score"], reverse=True)[:5]
    }

# ---------------- HEALTH CHECK ENDPOINT ----------------
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "video-moderation",
        "model": MODEL_NAME,
        "database": "connected" if client else "not connected",
        "video_processing": True,
        "timestamp": datetime.utcnow()
    }

# ---------------- VIDEO MODERATION ENDPOINT ----------------
@app.post("/predict-video")
async def predict_video(file: UploadFile = File(...)):

    start_time = time.time()

    status = "SUCCESS"
    video_duration = 0
    video_path = None

    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Only video files allowed")

    try:
        contents = await file.read()

        if not contents:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        # ----- STORE VIDEO IN SEPARATE GRIDFS BUCKET -----
        video_stream = io.BytesIO(contents)
        video_id = await video_fs_bucket.upload_from_stream(file.filename, video_stream)

        # Save temp file for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(contents)
            video_path = tmp.name

        # ----- GET VIDEO METADATA -----
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Invalid or corrupted video file")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        video_duration = round(frame_count / fps, 2) if fps else 0

        cap.release()

        # ----- EXTRACT FRAMES -----
        frames = extract_frames(video_path)

    except Exception as e:
        status = "FAILURE"

        end_time = time.time()
        processing_time = round(end_time - start_time, 3)

        error_metadata = {
            "filename": file.filename,
            "status": status,
            "error": str(e),
            "processing_time_seconds": processing_time,
            "video_duration_seconds": video_duration,
            "created_at": datetime.utcnow()
        }

        # Store error info in VIDEO collection
        await video_results_collection.insert_one(error_metadata)

        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if video_path and os.path.exists(video_path):
            os.remove(video_path)

    if not frames:
        raise HTTPException(status_code=400, detail="No readable frames found in video")

    unsafe_frames = 0
    safe_frames = 0

    unsafe_labels = set()
    safe_labels = set()

    for frame in frames:

        analysis = classify_image_full(frame)

        max_critical = analysis["max_critical_score"]
        max_unsafe = analysis["max_unsafe_score"]

        if max_critical >= 0.20:
            unsafe_frames += 1
            unsafe_labels.update(analysis["unsafe_labels"])
            continue

        if max_unsafe >= UNSAFE_THRESHOLD:
            unsafe_frames += 1
            unsafe_labels.update(analysis["unsafe_labels"])
        else:
            safe_frames += 1
            safe_labels.update(analysis["safe_labels"])

    total = len(frames)
    unsafe_percent = (unsafe_frames / total) * 100 if total else 0

    decision = "BLOCK" if unsafe_percent >= VIDEO_BLOCK_PERCENT else "SAFE"

    end_time = time.time()
    processing_time = round(end_time - start_time, 3)

    # ----- STORE VIDEO RESULT IN SEPARATE COLLECTION -----
    metadata = {
        "video_id": video_id,
        "filename": file.filename,
        "status": status,
        "decision": decision,
        "video_duration_seconds": video_duration,
        "processing_time_seconds": processing_time,
        "frames_analyzed": total,
        "unsafe_frames": unsafe_frames,
        "safe_frames": safe_frames,
        "unsafe_percentage": round(unsafe_percent, 2),
        "unsafe_labels": list(unsafe_labels),
        "safe_labels": list(safe_labels),
        "created_at": datetime.utcnow()
    }

    await video_results_collection.insert_one(metadata)

    return {
        "video_id": str(video_id),
        "filename": file.filename,
        "status": status,
        "video_duration_seconds": video_duration,
        "processing_time_seconds": processing_time,
        "frames_analyzed": total,
        "unsafe_frames": unsafe_frames,
        "safe_frames": safe_frames,
        "unsafe_percentage": round(unsafe_percent, 2),
        "decision": decision,
        "unsafe_labels_detected": list(unsafe_labels),
        "safe_labels_detected": list(safe_labels)
    }

# ---------------- SIMPLE DB TEST ENDPOINT ----------------
@app.get("/test-db")
async def test_db():
    return {"databases": await client.list_database_names()}
