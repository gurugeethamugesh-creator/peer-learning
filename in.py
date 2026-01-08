#!/usr/bin/env python3
"""
app.py - Single-file Flask app implementing:
Video -> Audio extraction (moviepy)
Transcription (whisper)
Important Notes / Key Points (transformers summarization pipeline)
TTS for Key Points (gTTS with pyttsx3 fallback)
Retrieval QA (sentence-transformers + sklearn NearestNeighbors)
SQLite logging (SQLAlchemy)
Simple Dashboard and Focus Timer logging
"""

import os
import re
import uuid
import io
import json
import datetime
from pathlib import Path
from functools import lru_cache

from flask import Flask, request, render_template, redirect, url_for, send_file, jsonify, flash
from werkzeug.utils import secure_filename

# ML / audio libs
from moviepy.editor import VideoFileClip
import whisper
from transformers import pipeline
from gtts import gTTS
import pyttsx3  # fallback if gTTS unavailable
import soundfile as sf
import numpy as np

# Embedding & indexing
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

# DB
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Config
UPLOAD_DIR = Path("uploads")
PROCESSED_DIR = Path("processed")
EMBED_DIR = Path("embeddings")
DB_PATH = "sqlite:///app.db"
ALLOWED_EXTENSIONS = {"mp4", "mov", "mkv", "webm", "avi", "mp3", "wav", "m4a"}

for d in (UPLOAD_DIR, PROCESSED_DIR, EMBED_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "supersecretkey")

# Database setup
Base = declarative_base()
engine = create_engine(DB_PATH, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)


class VideoSession(Base):
    __tablename__ = "video_sessions"
    id = Column(Integer, primary_key=True, index=True)
    uid = Column(String, unique=True, index=True)
    filename = Column(String)
    uploaded_at = Column(DateTime, default=datetime.datetime.utcnow)
    transcript = Column(Text)
    key_points = Column(Text)
    tts_mp3 = Column(String)
    embed_file = Column(String)
    duration_sec = Column(Float, default=0.0)
    total_focus_seconds = Column(Integer, default=0)


Base.metadata.create_all(bind=engine)


# -------------------------
# Lazy model loaders
# -------------------------
@lru_cache(maxsize=1)
def load_whisper_model(name="small"):
    print("Loading whisper model:", name)
    return whisper.load_model(name)


@lru_cache(maxsize=1)
def load_summarizer(model_name="sshleifer/distilbart-cnn-12-6"):
    print("Loading summarizer:", model_name)
    return pipeline("summarization", model=model_name, truncation=True)


@lru_cache(maxsize=1)
def load_sentence_model(model_name="all-MiniLM-L6-v2"):
    print("Loading sentence transformer:", model_name)
    return SentenceTransformer(model_name)


# -------------------------
# Utilities
# -------------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def simple_sentence_split(text):
    text = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r'(?<=[\.\?\!])\s+', text)
    parts = [p.strip() for p in parts if len(p.strip()) > 20]
    return parts if parts else [text]


def save_tts(text, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        tts = gTTS(text)
        tts.save(str(out_path))
        return str(out_path)
    except Exception as e:
        print("gTTS failed, trying pyttsx3 fallback:", e)
        try:
            engine = pyttsx3.init()
            engine.save_to_file(text, str(out_path))
            engine.runAndWait()
            return str(out_path)
        except Exception as e2:
            print("pyttsx3 fallback failed:", e2)
            return None


def embed_and_index(sentences, sess_uid):
    model = load_sentence_model()
    embeddings = model.encode(sentences, show_progress_bar=False)
    out_file = EMBED_DIR / f"{sess_uid}_embeddings.npz"
    np.savez_compressed(out_file, embeddings=embeddings, sentences=np.array(sentences, dtype=object))
    return str(out_file)


def load_embeddings(emb_file):
    data = np.load(emb_file, allow_pickle=True)
    return data["embeddings"], list(data["sentences"])


def nearest_context(question, emb_file, top_k=3):
    embeddings, sentences = load_embeddings(emb_file)
    s_model = load_sentence_model()
    q_emb = s_model.encode([question])
    nn = NearestNeighbors(n_neighbors=min(top_k, len(embeddings)))
    nn.fit(embeddings)
    dist, idx = nn.kneighbors(q_emb, return_distance=True)
    idx = idx[0]
    results = [{"sentence": sentences[i], "distance": float(dist[0][j])} for j, i in enumerate(idx)]
    return results


# -------------------------
# Flask routes
# -------------------------
@app.route("/")
def index():
    db = SessionLocal()
    sessions = db.query(VideoSession).order_by(VideoSession.uploaded_at.desc()).limit(10).all()
    return render_template("index.html", sessions=sessions)


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        flash("No file part")
        return redirect(url_for("index"))
    f = request.files["file"]
    if f.filename == "":
        flash("No selected file")
        return redirect(url_for("index"))
    if f and allowed_file(f.filename):
        filename = secure_filename(f.filename)
        uid = uuid.uuid4().hex
        save_path = UPLOAD_DIR / f"{uid}_{filename}"
        f.save(str(save_path))
        db = SessionLocal()
        vs = VideoSession(uid=uid, filename=filename)
        db.add(vs)
        db.commit()
        db.refresh(vs)
        try:
            process_upload(vs.uid, save_path)
            flash("File processed successfully")
        except Exception as e:
            flash(f"Processing error: {e}")
            print("Processing error:", e)
        return redirect(url_for("session_view", uid=uid))
    else:
        flash("File type not allowed")
        return redirect(url_for("index"))


def process_upload(uid, filepath: Path):
    db = SessionLocal()
    vs = db.query(VideoSession).filter(VideoSession.uid == uid).first()
    if not vs:
        raise RuntimeError("Session not found in DB")

    audio_out = PROCESSED_DIR / f"{uid}.wav"
    print("Extracting audio to", audio_out)
    clip = VideoFileClip(str(filepath))
    vs.duration_sec = clip.duration
    audio = clip.audio
    if audio is None:
        raise RuntimeError("No audio stream found in uploaded file.")
    audio.write_audiofile(str(audio_out), verbose=False, logger=None)
    clip.close()

    wmodel = load_whisper_model()
    print("Transcribing (this may take time)...")
    result = wmodel.transcribe(str(audio_out))
    transcript = result.get("text", "").strip()
    vs.transcript = transcript

    summarizer = load_summarizer()
    max_chunk = 1000
    chunks = [transcript[i:i+max_chunk] for i in range(0, len(transcript), max_chunk)]
    summaries = []
    for c in chunks:
        try:
            out = summarizer(c, max_length=130, min_length=30, do_sample=False)
            summaries.append(out[0]["summary_text"])
        except Exception as e:
            print("Summarizer chunk failed:", e)
            summaries.append(c[:500] + ("..." if len(c) > 500 else ""))
    key_points = "\n\n".join(summaries)
    vs.key_points = key_points

    tts_mp3_path = PROCESSED_DIR / f"{uid}_keypoints.mp3"
    tts_saved = save_tts(key_points, tts_mp3_path)
    vs.tts_mp3 = os.path.basename(tts_mp3_path) if tts_saved else None

    sentences = simple_sentence_split(transcript)
    emb_file = embed_and_index(sentences, uid)
    vs.embed_file = os.path.basename(emb_file)

    db.commit()
    db.close()
    print("Processing complete for", uid)


@app.route("/session/<uid>")
def session_view(uid):
    db = SessionLocal()
    vs = db.query(VideoSession).filter(VideoSession.uid == uid).first()
    if not vs:
        return "Session not found"
    return render_template("session.html", sess=vs)


@app.route("/download_tts/<uid>")
def download_tts(uid):
    db = SessionLocal()
    vs = db.query(VideoSession).filter(VideoSession.uid == uid).first()
    if not vs or not vs.tts_mp3:
        return "TTS not found", 404
    p = PROCESSED_DIR / vs.tts_mp3
    if not p.exists():
        return "TTS file missing", 404
    return send_file(str(p), mimetype="audio/mpeg", as_attachment=False, download_name=vs.tts_mp3)


@app.route("/ask/<uid>", methods=["POST"])
def ask(uid):
    data = request.get_json(force=True)
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "Missing question"}), 400

    db = SessionLocal()
    vs = db.query(VideoSession).filter(VideoSession.uid == uid).first()
    if not vs:
        return jsonify({"error": "Session not found"}), 404
    if not vs.embed_file:
        return jsonify({"error": "Embeddings not available"}), 500

    emb_file = EMBED_DIR / vs.embed_file
    if not emb_file.exists():
        return jsonify({"error": "Embeddings file missing on server"}), 500

    contexts = nearest_context(question, str(emb_file), top_k=4)
    context_text = "\n\n".join([c["sentence"] for c in contexts])

    prompt = f"Use the following extracted context from a lecture to answer the question concisely.\n\nContext:\n{context_text}\n\nQuestion: {question}\n\nAnswer (concise):"
    try:
        summarizer = load_summarizer()
        if len(prompt) > 1500:
            prompt = prompt[:1500]
        out = summarizer(prompt, max_length=120, min_length=20, do_sample=False)
        answer = out[0]["summary_text"]
    except Exception as e:
        print("Answer generation failed, fallback:", e)
        answer = contexts[0]["sentence"] if contexts else ""

    return jsonify({"answer": answer, "context": context_text})


@app.route("/log_focus", methods=["POST"])
def log_focus():
    data = request.get_json(force=True)
    uid = data.get("uid")
    secs = int(data.get("seconds", 0) or 0)
    if not uid:
        return jsonify({"error": "Missing uid"}), 400
    db = SessionLocal()
    vs = db.query(VideoSession).filter(VideoSession.uid == uid).first()
    if not vs:
        return jsonify({"error": "Session not found"}), 404
    vs.total_focus_seconds = (vs.total_focus_seconds or 0) + secs
    db.commit()
    return jsonify({"status": "ok", "total_focus_seconds": vs.total_focus_seconds})


@app.route("/dashboard")
def dashboard():
    db = SessionLocal()
    sessions = db.query(VideoSession).order_by(VideoSession.uploaded_at.desc()).all()
    total_sessions = len(sessions)
    total_focus = sum((s.total_focus_seconds or 0) for s in sessions)
    return render_template("dashboard.html", sessions=sessions, total_sessions=total_sessions, total_focus=total_focus)


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
