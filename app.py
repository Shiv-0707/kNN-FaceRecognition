#!/usr/bin/env python3
"""
app.py - Flask backend for Face Biometric Studio
(Minor changes: request landmarks from MTCNN; include x2/y2 and landmarks in faces JSON)
(Fix: crop faces from original PIL using detected boxes (with padding) before embedding so boxes and embeddings align.)
"""
import base64
import io
import json
import logging
import os
import pickle
import re
import threading
import time
import uuid
import zipfile
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file, abort

# Minimal: allow CORS for dev/debug if present
from flask_cors import CORS

import numpy as np
from PIL import Image
import cv2
import torch

# facenet-pytorch
try:
    from facenet_pytorch import MTCNN, InceptionResnetV1
except Exception as e:
    raise RuntimeError("facenet-pytorch is required. Install facenet-pytorch + torch.") from e

# ---------------- Config & storage root ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("faceapp")

BASE_DIR = Path(__file__).parent.resolve()
HF_PERSIST = Path("/persistent")
if HF_PERSIST.exists():
    STORAGE_ROOT = HF_PERSIST
else:
    STORAGE_ROOT = BASE_DIR

DATASET_DIR = STORAGE_ROOT / "dataset"
DATASET_DIR.mkdir(parents=True, exist_ok=True)
MODEL_META = STORAGE_ROOT / "embeddings_index.pkl"
PASSWORD_FILE = STORAGE_ROOT / "protected_passwords.json"

SAFE_NAME_RE = re.compile(r'[^A-Za-z0-9_\-]')
CROP_SIZE = (160, 160)
COSINE_THRESHOLD = 0.58  # matching threshold

device = torch.device("cpu")

logger.info("Initializing MTCNN and InceptionResnetV1 on device %s", device)
mtcnn = MTCNN(keep_all=True, device=device, post_process=True)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024

# ===== Minimal dev convenience: enable CORS for /api/* =====
CORS(app, resources={r"/api/*": {"origins": "*"}})

sessions = {}
sessions_lock = threading.Lock()
training_lock = threading.Lock()
training_status = {"running": False, "last_msg": "", "total": 0, "processed": 0, "percent": 0}

# Password store (plaintext as per your project)
def load_passwords_plain():
    if not PASSWORD_FILE.exists():
        return {}
    try:
        with open(PASSWORD_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_passwords_plain(data: dict):
    tmp = PASSWORD_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    tmp.replace(PASSWORD_FILE)

password_store = load_passwords_plain()

def set_plain_password_for(key: str, plain_password: str):
    password_store[key] = {"password": plain_password, "created": int(time.time())}
    save_passwords_plain(password_store)

def verify_plain_password_for(key: str, plain_password: str) -> bool:
    rec = password_store.get(key)
    if not rec:
        return False
    return rec.get("password") == plain_password

token_map = {}
token_lock = threading.Lock()

def make_token(key: str, ttl=300):
    tok = uuid.uuid4().hex
    with token_lock:
        token_map[tok] = {"key": key, "exp": time.time() + ttl}
    return tok

def verify_token(tok: str):
    with token_lock:
        rec = token_map.get(tok)
        if not rec:
            return None
        if time.time() > rec["exp"]:
            token_map.pop(tok, None)
            return None
        return rec["key"]

# ---------------- Helpers ----------------
def safe_name(name: str, uid: str):
    s = f"{name.strip()}_{uid}"
    return SAFE_NAME_RE.sub("_", s)[:60]

def decode_data_url(data_url: str):
    if not isinstance(data_url, str) or not data_url.startswith("data:image"):
        return None
    header, enc = data_url.split(",", 1)
    try:
        data = base64.b64decode(enc)
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None

def pil_from_bgr(img_bgr):
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

def tensor_to_pil(tensor):
    t = tensor.detach().cpu()
    t = (t + 1.0) / 2.0
    npimg = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(npimg)

def embed_from_pil(pil_img):
    try:
        from torchvision import transforms
        tf = transforms.Compose([
            transforms.Resize(CROP_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
        ])
        t = tf(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = resnet(t).cpu().numpy()[0]
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb.astype(np.float32)
    except Exception:
        logger.exception("embed_from_pil failed")
        return None

def extract_embedding(pil):
    try:
        # pil must be a face crop resized to CROP_SIZE
        return embed_from_pil(pil)
    except Exception:
        logger.exception("extract_embedding failed")
        return None

def cosine_similarity_vec(a, b):
    return float(np.dot(a, b))

def load_index():
    if MODEL_META.exists():
        try:
            with open(MODEL_META, "rb") as f:
                idx = pickle.load(f)
            return idx
        except Exception:
            logger.exception("Failed to load existing embeddings index")
    index = {}
    for person_dir in DATASET_DIR.iterdir():
        if not person_dir.is_dir():
            continue
        embs = []
        for p in person_dir.glob("*.npy"):
            try:
                e = np.load(str(p)).astype(np.float32)
                embs.append(e)
            except Exception:
                continue
        if embs:
            index[person_dir.name] = embs
    try:
        with open(MODEL_META, "wb") as f:
            pickle.dump(index, f)
    except Exception:
        logger.exception("Failed to save embeddings index")
    return index

def save_embedding(person_folder: Path, base_index: int, emb: np.ndarray):
    person_folder.mkdir(parents=True, exist_ok=True)
    fname = person_folder / f"{base_index:04d}.npy"
    np.save(str(fname), emb)

def save_image(person_folder: Path, base_index: int, pil_img: Image.Image):
    person_folder.mkdir(parents=True, exist_ok=True)
    fname = person_folder / f"{base_index:04d}.jpg"
    pil_img.save(str(fname), format="JPEG", quality=90)
    return fname

def crop_and_resize_face(pil_img: Image.Image, box, pad_frac=0.25):
    """
    Crop the PIL image using box=(x1,y1,x2,y2), add pad_frac fraction of width/height as padding,
    clip to image bounds, then resize to CROP_SIZE and return the PIL image.
    """
    try:
        x1, y1, x2, y2 = box
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        pad_w = int(round(w * pad_frac))
        pad_h = int(round(h * pad_frac))
        ix1 = max(0, x1 - pad_w)
        iy1 = max(0, y1 - pad_h)
        ix2 = min(pil_img.width, x2 + pad_w)
        iy2 = min(pil_img.height, y2 + pad_h)
        # ensure integer bounds
        ix1, iy1, ix2, iy2 = map(int, (ix1, iy1, ix2, iy2))
        crop = pil_img.crop((ix1, iy1, ix2, iy2))
        # Resize to CROP_SIZE preserving aspect (use ANTIALIAS)
        crop = crop.resize(CROP_SIZE, Image.LANCZOS)
        return crop
    except Exception:
        logger.exception("crop_and_resize_face failed")
        return None

# ---------------- Session cleanup ----------------
def session_cleanup_loop():
    while True:
        now = time.time()
        with sessions_lock:
            stale = [sid for sid, meta in sessions.items() if now - meta.get("last_active", now) > 60 * 5]
            for sid in stale:
                sessions.pop(sid, None)
        with token_lock:
            expired = [t for t, v in token_map.items() if time.time() > v["exp"]]
            for t in expired:
                token_map.pop(t, None)
        time.sleep(60)

cleanup_thread = threading.Thread(target=session_cleanup_loop, daemon=True)
cleanup_thread.start()

# ---------------- Routes ----------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/students/create", methods=["POST"])
def api_students_create():
    j = request.get_json(force=True)
    name = str(j.get("name", "")).strip()
    uid = str(j.get("uid") or str(uuid.uuid4())[:8]).strip()
    password = j.get("password")
    if not name or not uid or not password:
        return jsonify(ok=False, error="missing"), 400
    key = safe_name(name, uid)
    folder = DATASET_DIR / key
    folder.mkdir(parents=True, exist_ok=True)
    set_plain_password_for(key, password)
    return jsonify(ok=True, key=key, count=len(list(folder.glob("*.jpg"))))

@app.route("/api/students", methods=["GET"])
def api_students_list():
    out = []
    for p in DATASET_DIR.iterdir():
        if not p.is_dir(): continue
        name = p.name
        count = len(list(p.glob("*.jpg")))
        out.append({"key": name, "count": count})
    return jsonify(ok=True, students=out)

@app.route("/api/students/<key>/auth", methods=["POST"])
def api_students_auth(key):
    j = request.get_json(force=True)
    password = j.get("password")
    if not password:
        return jsonify(ok=False, error="missing"), 400
    if verify_plain_password_for(key, password):
        tok = make_token(key, ttl=300)
        return jsonify(ok=True, token=tok, expires_in=300)
    else:
        return jsonify(ok=False, error="bad_password"), 401

@app.route("/api/students/<key>/list", methods=["GET"])
def api_students_list_images(key):
    tok = request.args.get("token")
    if tok:
        good = verify_token(tok)
        if good != key:
            return jsonify(ok=False, error="invalid_token"), 403
    else:
        pwd = request.args.get("password") or (request.get_json(silent=True) or {}).get("password")
        if not pwd or not verify_plain_password_for(key, pwd):
            return jsonify(ok=False, error="auth_required"), 401
    folder = DATASET_DIR / key
    imgs = sorted([f.name for f in folder.glob("*.jpg")])[:2]
    return jsonify(ok=True, images=imgs)

@app.route("/api/students/<key>/image/<filename>", methods=["GET"])
def api_students_image_get(key, filename):
    tok = request.args.get("token")
    if tok:
        good = verify_token(tok)
        if good != key:
            return jsonify(ok=False, error="invalid_token"), 403
    else:
        pwd = request.args.get("password")
        if not pwd or not verify_plain_password_for(key, pwd):
            return jsonify(ok=False, error="auth_required"), 401
    folder = DATASET_DIR / key
    fpath = folder / filename
    if not fpath.exists() or not fpath.is_file():
        return jsonify(ok=False, error="not_found"), 404
    return send_file(str(fpath), mimetype="image/jpeg", as_attachment=False)

@app.route("/api/dataset/download", methods=["GET"])
def api_download():
    mem = io.BytesIO()
    try:
        with zipfile.ZipFile(mem, "w") as z:
            for d in DATASET_DIR.iterdir():
                if not d.is_dir():
                    continue
                for f in d.glob("*"):
                    z.write(f, f"{d.name}/{f.name}")
        mem.seek(0)
        return send_file(mem, mimetype="application/zip", as_attachment=True, download_name="dataset.zip")
    except Exception:
        logger.exception("zip failed")
        return jsonify(ok=False, error="zip_failed"), 500

@app.route("/api/enroll/start", methods=["POST"])
def enroll_start():
    j = request.get_json(silent=True)
    if not j:
        return jsonify(ok=False, error="invalid_json"), 400
    name = str(j.get("name", "unknown"))[:40].strip()
    uid = str(j.get("uid") or str(uuid.uuid4())[:8])
    try:
        target = int(j.get("target", 200))
        target = max(1, min(1000, target))
    except Exception:
        target = 200
    folder_name = safe_name(name, uid)
    folder = DATASET_DIR / folder_name
    folder.mkdir(parents=True, exist_ok=True)
    session_id = str(uuid.uuid4())
    with sessions_lock:
        sessions[session_id] = {"folder": folder, "count": len(list(folder.glob("*.jpg"))),
                                "name": name, "uid": uid, "target": target, "last_active": time.time(),
                                "inflight": 0}
    logger.info("Started enroll session %s for %s (target=%d)", session_id, folder_name, target)
    return jsonify(ok=True, session_id=session_id, folder=str(folder.name), count=sessions[session_id]["count"])

@app.route("/api/enroll/frame", methods=["POST"])
def enroll_frame():
    try:
        data = request.get_json(force=True)
    except Exception as e:
        logger.exception("Invalid json on enroll_frame")
        return jsonify(ok=False, error="invalid_json"), 400
    sid = data.get("session_id")
    image = data.get("image")
    if not sid or not image:
        return jsonify(ok=False, error="missing"), 400
    with sessions_lock:
        s = sessions.get(sid)
        if not s:
            return jsonify(ok=False, error="invalid_session"), 400
        if s.get("inflight", 0) > 6:
            return jsonify(ok=False, error="server_busy"), 429
        s["inflight"] = s.get("inflight", 0) + 1
        s["last_active"] = time.time()
    try:
        frame = decode_data_url(image)
        if frame is None:
            return jsonify(ok=False, error="bad_image"), 400
        if s["count"] >= s["target"]:
            return jsonify(ok=False, error="target_reached", count=s["count"]), 200
        pil = pil_from_bgr(frame)

        # DETECT boxes for frontend overlay + request landmarks (landmarks help alignment)
        try:
            boxes, probs, landmarks = mtcnn.detect(pil, landmarks=True)
        except Exception:
            logger.exception("MTCNN.detect failed")
            boxes, probs, landmarks = None, None, None

        # prepare boxes_json (include x2,y2 and landmarks)
        faces_json = []
        pil_faces_for_embedding = []
        if boxes is not None:
            for idx, b in enumerate(boxes):
                if b is None:
                    continue
                # round box and ensure ints
                x1, y1, x2, y2 = map(int, map(round, b))
                bw = max(0, x2 - x1)
                bh = max(0, y2 - y1)
                lm = None
                if landmarks is not None and idx < len(landmarks):
                    lm_pts = landmarks[idx].astype(float).tolist()
                    lm = [{"x": float(p[0]), "y": float(p[1])} for p in lm_pts]
                # Crop and resize from original PIL (ensures alignment between box and saved face)
                face_pil_crop = crop_and_resize_face(pil, (x1, y1, x2, y2))
                if face_pil_crop is None:
                    # fallback to using whole pil
                    face_pil_crop = pil.resize(CROP_SIZE, Image.LANCZOS)
                pil_faces_for_embedding.append(face_pil_crop)
                faces_json.append({
                    "x": x1, "y": y1, "x2": x2, "y2": y2,
                    "w": bw, "h": bh, "landmarks": lm
                })

        if not pil_faces_for_embedding:
            return jsonify(ok=True, count=s["count"], target=s["target"],
                           saved=False, duplicate=False, boxes=faces_json, message="no_face")

        # Use the first face crop for embedding (keeps behavior similar to before)
        face_pil = pil_faces_for_embedding[0]
        emb = extract_embedding(face_pil)
        if emb is None:
            logger.debug("embed failed or no face after alignment")
            return jsonify(ok=False, error="embed_failed"), 500

        folder = s["folder"]
        dup = False
        for p in folder.glob("*.npy"):
            try:
                old = np.load(str(p)).astype(np.float32)
                sim = cosine_similarity_vec(emb, old)
                if sim >= 0.92:
                    dup = True
                    break
            except Exception:
                continue
        if dup:
            return jsonify(ok=True, count=s["count"], target=s["target"], saved=False, duplicate=True, boxes=faces_json)

        next_idx = s.get("count", 0) + 1
        try:
            img_path = save_image(folder, next_idx, face_pil)
            save_embedding(folder, next_idx, emb)

            # IMMEDIATE INDEX UPDATE: append new embedding into embeddings_index.pkl
            try:
                if MODEL_META.exists():
                    with open(MODEL_META, "rb") as f:
                        idx = pickle.load(f)
                        if not isinstance(idx, dict):
                            idx = {}
                else:
                    idx = {}
                key_name = folder.name
                idx.setdefault(key_name, []).append(emb.astype(np.float32))
                with open(MODEL_META, "wb") as f:
                    pickle.dump(idx, f)
            except Exception:
                logger.exception("Failed to update embeddings index after enroll")
        except Exception:
            logger.exception("Failed saving enroll artifacts")
            return jsonify(ok=False, error="save_failed"), 500

        s["count"] = s.get("count", 0) + 1
        last_saved_url = None
        try:
            with open(img_path, "rb") as f:
                last_saved_url = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()
        except Exception:
            last_saved_url = None
        done = s["count"] >= s["target"]
        return jsonify(ok=True, count=s["count"], target=s["target"],
                       saved=True, duplicate=False, last_saved_url=None, done=done, boxes=faces_json)
    finally:
        with sessions_lock:
            s = sessions.get(sid)
            if s:
                s["inflight"] = max(0, s.get("inflight", 0) - 1)
                s["last_active"] = time.time()

@app.route("/api/enroll/stop", methods=["POST"])
def enroll_stop():
    j = request.get_json(force=True)
    sid = j.get("session_id")
    if not sid:
        return jsonify(ok=False, error="missing"), 400
    with sessions_lock:
        sessions.pop(sid, None)
    return jsonify(ok=True)

@app.route("/api/train", methods=["POST"])
def api_train():
    if training_status.get("running"):
        return jsonify(ok=False, message="training_already_running"), 429
    t = threading.Thread(target=training_thread, daemon=True)
    t.start()
    return jsonify(ok=True, message="training_started")

@app.route("/api/train/status", methods=["GET"])
def api_train_status():
    return jsonify(ok=True, status=training_status)

def prune_keep_two(person_dir: Path):
    emb_paths = sorted(person_dir.glob("*.npy"))
    if not emb_paths:
        return
    embs = []
    names = []
    for p in emb_paths:
        try:
            e = np.load(str(p)).astype(np.float32)
            embs.append(e)
            names.append(p.stem)
        except Exception:
            continue
    if not embs:
        return
    if len(embs) == 1:
        keep = [names[0]]
    else:
        best_pair = (0, 1)
        best_dist = -1.0
        for i in range(len(embs)):
            for j in range(i+1, len(embs)):
                sim = float(np.dot(embs[i], embs[j]))
                dist = 1.0 - sim
                if dist > best_dist:
                    best_dist = dist
                    best_pair = (i, j)
        keep = [names[best_pair[0]], names[best_pair[1]]]
    for p in person_dir.glob("*.npy"):
        stem = p.stem
        if stem not in keep:
            try:
                p.unlink()
            except Exception:
                logger.exception("Failed to delete embedding %s", p)
    for p in person_dir.glob("*.jpg"):
        stem = p.stem
        if stem not in keep:
            try:
                p.unlink()
            except Exception:
                logger.exception("Failed to delete image %s", p)

def training_thread():
    if training_lock.locked():
        return
    with training_lock:
        training_status.update({"running": True, "last_msg": "gathering", "total": 0, "processed": 0, "percent": 0})
        all_pairs = []
        for person_dir in DATASET_DIR.iterdir():
            if not person_dir.is_dir():
                continue
            for emb_path in person_dir.glob("*.npy"):
                all_pairs.append((person_dir.name, emb_path))
        total = len(all_pairs)
        training_status["total"] = total
        if total == 0:
            training_status.update({"running": False, "last_msg": "no_embeddings", "percent": 0})
            return
        index = {}
        for idx, (label, emb_path) in enumerate(all_pairs, start=1):
            training_status["processed"] = idx
            training_status["percent"] = int((idx / total) * 100)
            try:
                e = np.load(str(emb_path)).astype(np.float32)
                index.setdefault(label, []).append(e)
                training_status["last_msg"] = f"loaded:{emb_path.name}"
            except Exception:
                training_status["last_msg"] = f"skip:{emb_path.name}"
                continue
        try:
            # Merge with existing index to avoid clobbering live updates
            existing = {}
            if MODEL_META.exists():
                try:
                    with open(MODEL_META, "rb") as f:
                        existing = pickle.load(f) or {}
                except Exception:
                    existing = {}
            for k, v in index.items():
                existing.setdefault(k, []).extend(v)
            with open(MODEL_META, "wb") as f:
                pickle.dump(existing, f)
            training_status.update({"last_msg": f"trained_index_{len(existing)}", "percent": 100})
        except Exception:
            training_status.update({"last_msg": "save_failed", "percent": 0})
        for person_dir in DATASET_DIR.iterdir():
            if not person_dir.is_dir():
                continue
            try:
                prune_keep_two(person_dir)
            except Exception:
                logger.exception("prune failed for %s", person_dir)
        try:
            new_index = {}
            for person_dir in DATASET_DIR.iterdir():
                if not person_dir.is_dir():
                    continue
                embs = []
                for p in person_dir.glob("*.npy"):
                    try:
                        e = np.load(str(p)).astype(np.float32)
                        embs.append(e)
                    except Exception:
                        continue
                if embs:
                    new_index[person_dir.name] = embs
            with open(MODEL_META, "wb") as f:
                pickle.dump(new_index, f)
            training_status.update({"last_msg": "prune_and_rebuild_done", "percent": 100})
        except Exception:
            logger.exception("rebuild after prune failed")
        training_status["running"] = False

@app.route("/api/recognize/frame", methods=["POST"])
def api_recognize_frame():
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify(ok=False, error="invalid_json"), 400
    image = data.get("image")
    if not image:
        return jsonify(ok=False, error="missing"), 400
    frame = decode_data_url(image)
    if frame is None:
        return jsonify(ok=False, error="bad_image"), 400
    pil = pil_from_bgr(frame)
    h, w = frame.shape[:2]
    try:
        # request landmarks along with boxes
        boxes, probs, landmarks = mtcnn.detect(pil, landmarks=True)
    except Exception:
        logger.exception("MTCNN.detect failed")
        boxes, probs, landmarks = None, None, None
    # We will crop faces directly from detected boxes to ensure alignment
    faces_json = []
    pil_faces = []
    if boxes is not None:
        for idx, b in enumerate(boxes):
            if b is None:
                continue
            x1, y1, x2, y2 = map(int, map(round, b))
            bw = max(0, x2 - x1)
            bh = max(0, y2 - y1)
            xn = x1 / max(1, w); yn = y1 / max(1, h); wn = bw / max(1, w); hn = bh / max(1, h)
            lm = None
            if landmarks is not None and idx < len(landmarks):
                lm_pts = landmarks[idx].astype(float).tolist()
                lm = [{"x": float(p[0]), "y": float(p[1])} for p in lm_pts]
            faces_json.append({
                "x": x1, "y": y1, "x2": x2, "y2": y2,
                "w": bw, "h": bh, "xn": float(xn), "yn": float(yn), "wn": float(wn), "hn": float(hn),
                "landmarks": lm
            })
            # crop and resize for embedding/alignment using same method as enroll
            face_crop = crop_and_resize_face(pil, (x1, y1, x2, y2))
            if face_crop is not None:
                pil_faces.append(face_crop)
    index = load_index()
    if not pil_faces:
        return jsonify(ok=True, width=w, height=h, faces=faces_json)
    # For each cropped face derive embedding and match against index
    for i, face_pil in enumerate(pil_faces):
        emb = extract_embedding(face_pil)
        label = "Unknown"; prob = 0.0; matched = False
        if emb is not None and index:
            best_label = None; best_sim = -1.0
            for lab, embs in index.items():
                for e in embs:
                    sim = cosine_similarity_vec(emb, e)
                    if sim > best_sim:
                        best_sim = sim; best_label = lab
            if best_sim >= COSINE_THRESHOLD:
                label = best_label; prob = float(best_sim); matched = True
            else:
                label = "Unknown"; prob = float(best_sim); matched = False
        # attach label info to corresponding faces_json entry (safe check)
        if i < len(faces_json):
            faces_json[i]["label"] = label
            faces_json[i]["prob"] = prob
            faces_json[i]["matched"] = bool(matched)
        else:
            # append if mismatch in counts
            faces_json.append({
                "x": 0, "y": 0, "x2": 0, "y2": 0, "w": 0, "h": 0,
                "xn": 0.0, "yn": 0.0, "wn": 0.0, "hn": 0.0,
                "label": label, "prob": prob, "matched": bool(matched),
                "landmarks": None
            })
    return jsonify(ok=True, width=w, height=h, faces=faces_json)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)
