import os
import cv2
import numpy as np
import faiss
import logging
import base64
import requests
from datetime import datetime
from typing import List, Dict, Tuple
import random
import insightface
import sqlite3
import re
import subprocess
import sys
from flask import (
    Flask, request, render_template, session,
    redirect, url_for, flash
)
from google_auth_oauthlib.flow import Flow
from google.oauth2 import id_token

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)   # or CRITICAL

# Relax scope checking for Google's alias → full URL expansion
os.environ['OAUTHLIB_RELAX_TOKEN_SCOPE'] = '1'

# Allow insecure transport only in development
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'  # Remove in production!
# ========================
# Configuration & Paths
# ========================

# Base directory (must be defined early)
if os.getenv('RENDER') == 'true':
    base_dir = '/app'
else:
    base_dir = os.path.dirname(os.path.abspath(__file__))

# Database
DB_PATH = os.path.join(base_dir, "data", "users.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
contributions_dir = os.path.join(base_dir, "data", "contributions")
os.makedirs(contributions_dir, exist_ok=True)
# Paths
features_file = "/content/drive/MyDrive/machines/npys/features3_fusion_merged.npy"
incremental_features_file = os.getenv(
    "INCREMENTAL_FEATURES_FILE",
    "/content/drive/MyDrive/machines/npys/features3_fusion_new.npy",
)
INCREMENTAL_FALLBACK_DISTANCE = float(os.getenv("INCREMENTAL_FALLBACK_DISTANCE", "1.0"))
AUTO_UPDATE_INCREMENTAL_FEATURES = os.getenv("AUTO_UPDATE_INCREMENTAL_FEATURES", "1").strip() not in {"0", "false", "False"}
UPDATE_FEATURES_SCRIPT = os.getenv("UPDATE_FEATURES_SCRIPT", "/content/drive/MyDrive/app/update_incremental_features.py")
report_file = os.path.join(base_dir, "data", "search_report.txt")
uploads_dir = os.path.join(base_dir, "data", "uploads")
db_txt = "/content/drive/MyDrive/dev/txt/data/db.txt"
saved_txt = "/content/drive/MyDrive/dev/txt/data/insta.txt"
SUPER_USER_EMAIL = "shihabaaqil2224@gmail.com"  # Change to your email
NGROK_URL="https://nonrealistically-techy-quinn.ngrok-free.app"

os.makedirs(uploads_dir, exist_ok=True)

# Load db.txt and saved.txt at startup
try:
    with open(db_txt, 'r', encoding='utf-8') as f:
        db_lines = f.readlines()
    with open(saved_txt, 'r', encoding='utf-8') as f:
        saved_lines = f.readlines()
except FileNotFoundError as e:
    logging.error(f"Database file not found: {e}")
    db_lines = saved_lines = []

# ========================
# SQLite Setup
# ========================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            name TEXT,
            points INTEGER DEFAULT 0,
            joined_date TEXT DEFAULT CURRENT_TIMESTAMP,
            last_login TEXT
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS contributions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            person_name TEXT,
            image_count INTEGER,
            points_awarded INTEGER,
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(user_id)
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# ========================
# Flask App
# ========================
app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'dev-secret-key-change-in-production')

# Google OAuth
CLIENT_SECRETS_FILE = "/content/drive/MyDrive/app/client_secrets.json"
SCOPES = ['openid', 'email', 'profile']

# Allow HTTP in dev
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'  # Remove in production!

def _get_redirect_uri():
    # Priority 1: explicit env URL (recommended for ngrok/deploy)
    ngrok_url = os.getenv('NGROK_URL', '').strip()
    if ngrok_url:
        return ngrok_url.rstrip('/') + '/callback'

    # Priority 2: current request host (works for local + proxy hosts)
    proto = request.headers.get('X-Forwarded-Proto', request.scheme)
    host = request.headers.get('X-Forwarded-Host', request.host)
    return f"{proto}://{host}/callback"


def _build_google_flow(redirect_uri: str):
    return Flow.from_client_secrets_file(
        CLIENT_SECRETS_FILE,
        scopes=SCOPES,
        redirect_uri=redirect_uri
    )

try:
    face_analysis = insightface.app.FaceAnalysis(
        det_name='retinaface_r50_v1',
        rec_name='arcface_r100_v1'
    )
    face_analysis.prepare(ctx_id=-1, det_size=(640, 640))          # ← MUST match training script
    print("InsightFace old models (retinaface_r50 + arcface_r100) loaded OK")
except Exception as e:
    print("Failed to load retinaface_r50_v1 + arcface_r100_v1:")
    print(str(e))
    print("\nThis usually means your insightface version is too new.")
    print("Try:   pip install insightface==0.7.3   or   ==0.7.2   or   ==0.6.2")
    raise   # stop the app so you notice immediately

# FAISS cache
global_index = None
global_image_paths = None
global_incremental_index = None
global_incremental_image_paths = None
global_incremental_mtime = None

def extract_facial_features(image: np.ndarray, image_path: str) -> Dict:
    try:
        faces = face_analysis.get(image)
        if not faces:
            top, bottom, left, right = 250, 250, 250, 250
            color = (0, 0, 0)
            padded = cv2.copyMakeBorder(
                image,
                top, bottom, left, right,
                borderType=cv2.BORDER_CONSTANT,
                value=color
            )
            faces = face_analysis.get(padded)
            if not faces:
                logging.warning(f"No faces detected in {image_path}")
                return {'features': None, 'image_path': os.path.basename(image_path), 'status': 'failed','reason': 'No faces detected', 'detection_method': 'insightface', 'confidence': 0.0}
        # Prefer female faces if present
        female_faces = [f for f in faces if hasattr(f, "gender") and f.gender == 0]
        if female_faces:
            faces = female_faces
        areas = [(f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]) for f in faces]
        max_idx = np.argmax(areas)
        selected_face = faces[max_idx]
        embedding = selected_face.embedding / np.linalg.norm(selected_face.embedding)
        confidence = selected_face.det_score*0.8
        #logging.warning(f"Landmarks missing or invalid for {image_path}, using default features")
        return {'features': embedding, 'image_path': os.path.basename(image_path), 'status': 'success', 'reason': None, 'detection_method': 'insightface', 'confidence': confidence}
    except Exception as e:
        logging.error(f"Feature extraction failed for {image_path}: {str(e)}")
        return {'features': None, 'image_path': os.path.basename(image_path), 'status': 'failed', 'reason': f'Exception: {str(e)}', 'detection_method': 'insightface', 'confidence': 0.0}

def load_existing_features(features_file: str) -> Tuple[List[np.ndarray], List[str]]:
    try:
        if os.path.exists(features_file):
            data = np.load(features_file, allow_pickle=True).item()
            image_paths = data.get('image_paths', [])
            feature_vectors = data.get('feature_vectors', [])
            # Convert full paths to filenames for consistency
            image_paths = [os.path.basename(path) if os.path.sep in path else path for path in image_paths]
            return feature_vectors, image_paths
        return [], []
    except Exception as e:
        logging.error(f"Failed to load existing features: {str(e)}")
        return [], []

def build_feature_index(feature_vectors: List[np.ndarray]) -> faiss.Index:
    try:
        if not feature_vectors:
            raise ValueError("Feature vector list is empty")
        dimension = len(feature_vectors[0])
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(feature_vectors).astype(np.float32))
        return index
    except Exception as e:
        logging.error(f"Failed to build FAISS index: {str(e)}")
        raise

def reverse_image_search(query_image_path: str, index: faiss.Index, image_paths: List[str], top_k: int = 5) -> Tuple[List[Dict], str]:
    try:
        query_image = cv2.imread(query_image_path)
        if query_image is None:
            raise ValueError(f"Failed to load query image: {query_image_path}")
        query_result = extract_facial_features(query_image, query_image_path)
        if query_result['status'] == 'failed':
            raise ValueError(f"Failed to extract features from query image: {query_result['reason']}")
        query_vector = query_result['features'].reshape(1, -1).astype(np.float32)
        if query_vector.shape[1] != index.d:
            raise ValueError(f"Query vector dimension {query_vector.shape[1]} does not match index dimension {index.d}")
        distances, indices = index.search(query_vector, top_k)
        results = []
        for i in range(top_k):
            idx = indices[0][i]
            if idx < len(image_paths):
                results.append({
                    'image_path': image_paths[idx],
                    'distance': float(distances[0][i]),
                    'index': int(idx)
                })
        return results, query_result['detection_method']
    except Exception as e:
        logging.error(f"Reverse image search failed for {query_image_path}: {str(e)}")
        raise

def _search_index(query_vector: np.ndarray, index: faiss.Index, image_paths: List[str], top_k: int, source: str):
    if index is None or not image_paths:
        return []
    distances, indices = index.search(query_vector, top_k)
    results = []
    for i in range(top_k):
        idx = indices[0][i]
        if 0 <= idx < len(image_paths):
            results.append({
                'image_path': image_paths[idx],
                'distance': float(distances[0][i]),
                'index': int(idx),
                'source': source,
            })
    return results


def reverse_image_search_multi(
    query_image_path: str,
    primary_index: faiss.Index,
    primary_paths: List[str],
    incremental_index: faiss.Index = None,
    incremental_paths: List[str] = None,
    top_k: int = 5,
) -> Tuple[List[Dict], str]:
    try:
        query_image = cv2.imread(query_image_path)
        if query_image is None:
            raise ValueError(f"Failed to load query image: {query_image_path}")
        query_result = extract_facial_features(query_image, query_image_path)
        if query_result['status'] == 'failed':
            raise ValueError(f"Failed to extract features from query image: {query_result['reason']}")
        query_vector = query_result['features'].reshape(1, -1).astype(np.float32)

        primary_results = []
        if primary_index is not None:
            if query_vector.shape[1] != primary_index.d:
                raise ValueError(f"Query vector dimension {query_vector.shape[1]} does not match primary index dimension {primary_index.d}")
            primary_results = _search_index(query_vector, primary_index, primary_paths, top_k, "primary")

        # Fast path: strong primary match found, skip incremental search.
        if primary_results:
            best_primary = min(r['distance'] for r in primary_results)
            if best_primary < INCREMENTAL_FALLBACK_DISTANCE:
                return sorted(primary_results, key=lambda x: x['distance'])[:top_k], query_result['detection_method']

        combined_results = list(primary_results)
        if incremental_index is not None:
            if query_vector.shape[1] == incremental_index.d:
                combined_results.extend(
                    _search_index(query_vector, incremental_index, incremental_paths or [], top_k, "incremental")
                )
            else:
                logging.warning(
                    "Skipping incremental index due to dim mismatch: query=%s incremental=%s",
                    query_vector.shape[1],
                    incremental_index.d,
                )

        best_by_path = {}
        for r in combined_results:
            p = r['image_path']
            if p not in best_by_path or r['distance'] < best_by_path[p]['distance']:
                best_by_path[p] = r
        final_results = sorted(best_by_path.values(), key=lambda x: x['distance'])[:top_k]
        return final_results, query_result['detection_method']
    except Exception as e:
        logging.error(f"Reverse image search (multi) failed for {query_image_path}: {str(e)}")
        raise


def load_and_cache_indexes():
    global global_index, global_image_paths
    global global_incremental_index, global_incremental_image_paths, global_incremental_mtime

    if global_index is None:
        feature_vectors, image_paths = load_existing_features(features_file)
        global_index = build_feature_index(feature_vectors)
        global_image_paths = image_paths
        logging.info("Loaded primary index with %s vectors", len(image_paths))

    if os.path.exists(incremental_features_file):
        try:
            mtime = os.path.getmtime(incremental_features_file)
            if global_incremental_mtime != mtime:
                inc_vectors, inc_paths = load_existing_features(incremental_features_file)
                if inc_vectors:
                    global_incremental_index = build_feature_index(inc_vectors)
                    global_incremental_image_paths = inc_paths
                    global_incremental_mtime = mtime
                    logging.info("Reloaded incremental index with %s vectors", len(inc_paths))
                else:
                    global_incremental_index = None
                    global_incremental_image_paths = []
                    global_incremental_mtime = mtime
        except Exception as e:
            logging.error(f"Failed to load incremental index: {e}")
            global_incremental_index = None
            global_incremental_image_paths = []
    else:
        global_incremental_index = None
        global_incremental_image_paths = []
        global_incremental_mtime = None

    return global_index, global_image_paths, global_incremental_index, global_incremental_image_paths

# Fixed main() - now takes query_path directly
def main(query_image_path: str, top_k: int = 30):
    try:
        index, image_paths, inc_index, inc_paths = load_and_cache_indexes()
        results, detection_method = reverse_image_search_multi(
            query_image_path,
            index,
            image_paths,
            incremental_index=inc_index,
            incremental_paths=inc_paths,
            top_k=top_k,
        )
        return results, detection_method
    except Exception as e:
        logging.error(f"Main execution failed: {str(e)}")
        raise

# ========================
# Points System
# ========================
def award_points(user_id: str, points: int, person_name: str = None, image_count: int = 0):
    if points <= 0:
        return False
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE users SET points = points + ? WHERE user_id = ?", (points, user_id))
    if person_name:
        c.execute(
            "INSERT INTO contributions (user_id, person_name, image_count, points_awarded) "
            "VALUES (?, ?, ?, ?)",
            (user_id, person_name, image_count, points)
        )
    conn.commit()
    conn.close()

    # Update session if current user
    if session.get('user_id') == user_id:
        session['points'] = session.get('points', 0) + points
    return True


def auto_update_incremental_feature(saved_image_path: str, image_id: str):
    if not AUTO_UPDATE_INCREMENTAL_FEATURES:
        return False, "auto update disabled"
    if not os.path.exists(UPDATE_FEATURES_SCRIPT):
        return False, f"script not found: {UPDATE_FEATURES_SCRIPT}"

    cmd = [sys.executable, UPDATE_FEATURES_SCRIPT, saved_image_path, image_id]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if proc.returncode == 0:
            out = (proc.stdout or "").strip()
            return True, out or "incremental features updated"
        err = (proc.stderr or proc.stdout or "").strip()
        return False, err or f"feature update failed with code {proc.returncode}"
    except Exception as e:
        return False, str(e)

# ========================
# Suggested Names from missminimized.csv
# Format: ID,main_name/alias1/alias2/...
# ========================
SUGGESTED_TXT = '/content/drive/MyDrive/dev/csv/missminimized.csv'
BLANK_TXT = '/content/drive/MyDrive/dev/csv/blank.csv'

def load_suggested_names() -> List[Dict]:
    if not is_super_user():
        return "Access denied", 403
    """Load suggestions: ID, main_name, list of aliases"""
    suggestions = []
    if not os.path.exists(SUGGESTED_TXT):
        #logging.error(f"Suggested file not found: {SUGGESTED_TXT}")
        return suggestions

    try:
        with open(SUGGESTED_TXT, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or ',' not in line:
                    continue
                id_part, names_part = line.split(',', 1)
                id_str = id_part.strip()
                all_names = [n.strip() for n in names_part.split('/') if n.strip()]
                feedback = None
                for i in range(len(all_names)):
                    if all_names[i] == 'notfound':
                        feedback = 'notfound'
                        break
                    elif all_names[i] == 'banned':
                        feedback = 'banned'
                        break
                    elif all_names[i] == 'common':
                        feedback = 'common'
                        break
                    else:
                        feedback = None
                if not all_names:
                    continue
                main_name = all_names[0]  # First is main folder name
                aliases = all_names[1:]   # Rest are aliases
                # All names (main + aliases) for matching
                all_search_names = [main_name] + aliases
                suggestions.append({
                    'id': id_str,
                    'main_name': main_name,
                    'aliases': aliases,
                    'all_names': [n.lower() for n in all_search_names],  # for matching
                    'feedback': feedback,
                    'search_names': names_part.strip()
                })
    except Exception as e:
        logging.error(f"Failed to load suggested names: {e}")

    #logging.info(f"Loaded {len(suggestions)} suggested profiles")
    return suggestions


def load_blank_map() -> Dict[str, str]:
    blank_map = {}
    if not os.path.exists(BLANK_TXT):
        return blank_map
    try:
        with open(BLANK_TXT, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or "," not in s:
                    continue
                id_part, names_part = s.split(",", 1)
                slot_id = id_part.strip()
                names_part = names_part.strip()
                if not slot_id or not names_part:
                    continue
                # keep main name before aliases/metadata
                main_name = names_part.split("/", 1)[0].strip()
                if main_name:
                    blank_map[slot_id] = main_name
    except Exception as e:
        logging.warning(f"Failed to load blank map: {e}")
    return blank_map

def get_random_suggestion() -> Dict | None:
    if not is_super_user():
        return "Access denied", 403
    """Get one random pending suggestion (no feedback yet)"""
    all_suggestions = load_suggested_names()
    if not all_suggestions:
        return None

    # Only show suggestions without feedback
    pending = [s for s in all_suggestions if s['feedback'] is None]
    if not pending:
        return None

    choice = random.choice(pending)
    google_link = f"https://www.google.com/search?tbm=isch&q={choice['search_names'].replace(' ', '+').replace('/', '%2F')}+onlyfans"

    return {
        'id': choice['id'],
        'name': choice['main_name'],  # Show main name
        'link': google_link,
        'feedback': None
    }

def remove_suggested_name(target_id: str):
    if not is_super_user():
        return "Access denied", 403
    """Remove suggestion by ID after successful contribution"""
    all_suggestions = load_suggested_names()
    updated = [s for s in all_suggestions if s['id'] != target_id]

    if len(updated) == len(all_suggestions):
        #logging.warning(f"Suggested ID not found for removal: {target_id}")
        return

    try:
        with open(SUGGESTED_TXT, 'w', encoding='utf-8') as f:
            for s in updated:
                aliases_str = '/'.join(s['aliases']) if s['aliases'] else ''
                line = f"{s['id']},{s['main_name']}"
                if aliases_str:
                    line += f"/{aliases_str}"
                f.write(line + '\n')
        #logging.info(f"Removed suggested ID: {target_id}")
    except Exception as e:
        logging.error(f"Failed to remove suggestion {target_id}: {e}")

def save_feedback(name: str, feedback: str):
    if not is_super_user():
        return "Access denied", 403
    """Save feedback (banned/common/notfound) only once per profile"""
    if feedback not in {'banned', 'common', 'notfound'}:
        #logging.error(f"Invalid feedback value: {feedback}")
        return

    suggestions = load_suggested_names()
    updated = []
    found = False

    for s in suggestions:
        # Match if the submitted name equals main_name or any alias (case-insensitive)
        if name.lower() in s['all_names']:
            if s['feedback'] is None:  # Only add feedback if not already set
                s['feedback'] = feedback
                # Append feedback once at the end
                s['search_names'] = s['search_names'] + f"/{feedback}"
                found = True
            # If already has feedback, keep it unchanged
        updated.append(s)

    # If name not found at all (shouldn't happen normally)
    if not found:
        #logging.warning(f"Feedback for unknown name: {name}")
        updated.append({
            'id': 'unknown',
            'main_name': name,
            'aliases': [],
            'all_names': [name.lower()],
            'feedback': feedback,
            'search_names': f"{name}/{feedback}"
        })

    # Write back the file
    try:
        with open(SUGGESTED_TXT, 'w', encoding='utf-8') as f:
            for s in updated:
                f.write(f"{s['id']},{s['search_names']}\n")
        #logging.info(f"Feedback saved: '{name}' → {feedback}")
    except Exception as e:
        #logging.error(f"Failed to save feedback: {e}")
        raise

def refresh_user_points():
    if 'user_id' not in session:
        return
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT points FROM users WHERE user_id = ?", (session['user_id'],))
    row = c.fetchone()
    conn.close()
    if row:
        session['points'] = row[0]
    else:
        session['points'] = 0

# ========================
# Auth Routes
# ========================

# Make is_super_user available in ALL templates
@app.context_processor
def utility_processor():
    return dict(
        is_super_user = lambda: session.get('user_email') == SUPER_USER_EMAIL
    )

@app.route('/login')
def login():
    redirect_uri = _get_redirect_uri()
    #logging.warning(f"[OAuth] /login using redirect_uri={redirect_uri}")
    flow = _build_google_flow(redirect_uri)
    authorization_url, state = flow.authorization_url(
        access_type='offline',
        include_granted_scopes='true'
    )
    session['state'] = state
    session['oauth_redirect_uri'] = redirect_uri
    # Persist PKCE verifier so callback can exchange code successfully.
    if getattr(flow, "code_verifier", None):
        session['oauth_code_verifier'] = flow.code_verifier
    return redirect(authorization_url)

@app.route('/callback')
def callback():
    if request.args.get('state') != session.get('state'):
        return "State mismatch!", 400

    try:
        redirect_uri = session.get('oauth_redirect_uri') or _get_redirect_uri()
        logging.warning(f"[OAuth] /callback using redirect_uri={redirect_uri}")
        flow = _build_google_flow(redirect_uri)
        code_verifier = session.get('oauth_code_verifier')
        if code_verifier:
            flow.fetch_token(authorization_response=request.url, code_verifier=code_verifier)
        else:
            flow.fetch_token(authorization_response=request.url)
        credentials = flow.credentials

        from google.auth.transport.requests import Request as GoogleAuthRequest
        google_request = GoogleAuthRequest()
        id_info = id_token.verify_oauth2_token(credentials.id_token, google_request)

        user_id = id_info['sub']
        email = id_info.get('email')
        name = id_info.get('name', email.split('@')[0] if email else 'User')

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        # Try to insert — if it already exists, rowcount == 0
        c.execute(
            "INSERT OR IGNORE INTO users (user_id, email, name, points, joined_date) "
            "VALUES (?, ?, ?, 5, CURRENT_TIMESTAMP)",
            (user_id, email, name)
        )

        was_new_user = c.rowcount > 0   # True only on first signup

        # Always update name and last_login (in case name changed)
        c.execute(
            "UPDATE users SET name = ?, last_login = ? WHERE user_id = ?",
            (name, datetime.now().isoformat(), user_id)
        )

        # Get current points
        c.execute("SELECT points FROM users WHERE user_id = ?", (user_id,))
        row = c.fetchone()
        points = row[0] if row else 0

        conn.commit()
        conn.close()

        # Store in session
        session['user_id'] = user_id
        session['user_email'] = email
        session['user_name'] = name
        session['points'] = points

        # Welcome message — different for new vs returning users
        if was_new_user:
            flash(f"Welcome, {name}! You've been given 5 starting points 🎉", "success")
        else:
            flash(f"Welcome back, {name}! You have {points} points.", "success")

    except Exception as e:
        logging.exception("[OAuth] Login failed")
        flash(f"Login failed: {str(e)}", "error")
        return redirect(url_for('index'))

    return redirect(url_for('index'))

@app.route('/logout')
def logout():
    session.clear()
    flash("Logged out successfully.", "info")
    return redirect(url_for('index'))

def login_required(f):
    def wrapper(*args, **kwargs):
        if 'user_id' not in session:
            flash("Please log in first.", "warning")
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    wrapper.__name__ = f.__name__  # Preserve name for Flask
    return wrapper

# ========================
# Main Routes
# ========================
@app.route('/')
def index():
    refresh_user_points()
    return render_template('index.html')

@app.route('/upload', methods=['GET'])
@login_required
def upload_form():
    return render_template('upload.html')

@app.route('/feedback', methods=['POST'])
@login_required
def feedback():
    if not is_super_user():
        return "Access denied", 403
    name = request.form.get('name')
    choice = request.form.get('choice')  # 'banned', 'common', 'notfound'

    if not name or choice not in ['banned', 'common', 'notfound']:
        flash("Invalid feedback.", "error")
        return redirect(url_for('contribute'))

    save_feedback(name.strip(), choice)
    award_points(session['user_id'], 1, person_name=f"Feedback on '{name}'", image_count=0)
    flash(f"Thanks for feedback on '{name}'! +1 point 🎉", "success")
    return redirect(url_for('contribute'))

def is_super_user():
    return session.get('user_email') == SUPER_USER_EMAIL

from flask import make_response

@app.route('/admin')
@login_required
def admin_dashboard():
    if not is_super_user():
        flash("Access denied. Super user only.", "error")
        return redirect(url_for('index'))
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT user_id, email, name, points FROM users ORDER BY points DESC")
    users = c.fetchall()
    c.execute("""SELECT c.timestamp, u.email, u.name, c.person_name, c.image_count, c.points_awarded
                 FROM contributions c
                 JOIN users u ON c.user_id = u.user_id
                 ORDER BY c.timestamp DESC LIMIT 50""")
    recent_contribs = c.fetchall()
    conn.close()

    resp = make_response(render_template('admin.html', users=users, recent_contribs=recent_contribs))
    
    # Prevent browser from caching this page
    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    
    return resp

@app.route('/admin/edit_points', methods=['POST'])
@login_required
def admin_edit_points():
    if not is_super_user():
        return "Access denied", 403
    
    user_id    = request.form.get('user_id')
    action     = request.form.get('action')
    amount_str = request.form.get('amount', '').strip()

    #logging.info(f"EDIT_POINTS called | user_id={user_id} action={action} amount={amount_str}")

    if not user_id or not action:
        flash("Invalid request.", "error")
        return redirect(url_for('admin_dashboard'))

    try:
        amount = int(amount_str) if amount_str else 0
        if amount < 0 and action != 'reset':
            raise ValueError("Negative amount not allowed except reset")
    except Exception as e:
        #logging.error(f"Amount parsing failed: {e}")
        flash("Invalid amount.", "error")
        return redirect(url_for('admin_dashboard'))

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    try:
        if action == 'add':
            c.execute("UPDATE users SET points = points + ? WHERE user_id = ?", (amount, user_id))
            #logging.info(f"Added {amount} to {user_id}")
        elif action == 'subtract':
            c.execute("""
                UPDATE users
                SET points = CASE
                    WHEN points - ? < 0 THEN 0
                    ELSE points - ?
                END
                WHERE user_id = ?
            """, (amount, amount, user_id))
            #logging.info(f"Subtracted {amount} from {user_id}")
        elif action == 'reset':
            c.execute("UPDATE users SET points = 0 WHERE user_id = ?", (user_id,))
            #logging.info(f"Reset points for {user_id}")
        else:
            raise ValueError(f"Unknown action: {action}")

        if c.rowcount == 0:
            #logging.warning(f"No rows affected for user_id={user_id} - maybe doesn't exist?")
            flash("No user found with that ID.", "warning")
        else:
            flash(f"{action.title()} successful.", "success")

        conn.commit()
    except Exception as e:
        conn.rollback()
        #logging.error(f"Database error during edit: {e}")
        flash("Database error during edit.", "error")
    finally:
        conn.close()

    return redirect(url_for('admin_dashboard'))

@app.route('/admin/delete_user', methods=['POST'])
@login_required
def admin_delete_user():
    if not is_super_user():
        return "Access denied", 403

    user_id = request.form.get('user_id')
    if not user_id:
        flash("No user selected.", "error")
        return redirect(url_for('admin_dashboard'))

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM contributions WHERE user_id = ?", (user_id,))
    c.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()

    flash("User deleted.", "success")
    return redirect(url_for('admin_dashboard'))

@app.route('/contribute', methods=['GET', 'POST'])
@login_required
def contribute():
    if not is_super_user():
        return "Access denied", 403
    if request.method == 'GET':
        suggestion = get_random_suggestion()
        return render_template('contribute.html', suggestion=suggestion)

    person_name = request.form.get('person_name', '').strip()
    suggestion_id = request.form.get('suggestion_id', '').strip()
    images = request.files.getlist('images')

    if not images or all(not f.filename for f in images):
        flash("Please select at least one image.", "error")
        return redirect(url_for('contribute'))

    if not suggestion_id:
        flash("No active suggestion. Please refresh to get a new one.", "error")
        return redirect(url_for('contribute'))

    # Load and validate suggestion by ID
    suggested_list = load_suggested_names()
    matched_suggestion = next((s for s in suggested_list if s['id'] == suggestion_id), None)

    if not matched_suggestion:
        flash("Invalid or expired suggestion. Please refresh.", "error")
        return redirect(url_for('contribute'))

    # CRITICAL: Enforce exact name match (case-insensitive)
    if person_name.lower() != matched_suggestion['main_name'].lower():
        flash(
            f"Name mismatch! You must use exactly: <strong>{matched_suggestion['main_name']}</strong><br>"
            f"Do not edit the name. Refresh for a new suggestion if needed.",
            "error"
        )
        return redirect(url_for('contribute'))

    # All checks passed — proceed
    folder_name = matched_suggestion['id']
    display_name = matched_suggestion['main_name']
    remove_id = matched_suggestion['id']

    profile_folder = os.path.join(base_dir, "data", "contributions", folder_name)
    os.makedirs(profile_folder, exist_ok=True)

    existing_files = [f for f in os.listdir(profile_folder)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
    existing_count = len(existing_files)

    MAX_IMAGES_PER_PERSON = 10
    remaining_slots = MAX_IMAGES_PER_PERSON - existing_count

    if remaining_slots <= 0:
        flash(f"Maximum {MAX_IMAGES_PER_PERSON} images already reached for '{display_name}'.", "error")
        return redirect(url_for('contribute'))

    new_images = images[:remaining_slots]
    if len(images) > remaining_slots:
        flash(f"Only {remaining_slots} new images accepted (max {MAX_IMAGES_PER_PERSON} total).", "warning")

    saved_count = 0
    saved_paths = []
    for file in new_images:
        if file.filename and file.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            save_path = os.path.join(profile_folder, file.filename)
            if os.path.exists(save_path):
                base, ext = os.path.splitext(file.filename)
                counter = 1
                while os.path.exists(save_path):
                    save_path = os.path.join(profile_folder, f"{base}_{counter}{ext}")
                    counter += 1
            file.save(save_path)
            saved_count += 1
            saved_paths.append(save_path)

    if saved_count == 0:
        flash("No valid images were uploaded.", "error")
        return redirect(url_for('contribute'))

    points = saved_count * 1
    award_points(session['user_id'], points, display_name, saved_count)
    feature_ok_count = 0
    feature_fail_count = 0
    for path in saved_paths:
        ok, msg = auto_update_incremental_feature(path, os.path.basename(path))
        if ok:
            feature_ok_count += 1
        else:
            feature_fail_count += 1
            logging.warning(f"Incremental feature update failed for {path}: {msg}")

    remove_suggested_name(remove_id)
    flash(f"Success! Added {saved_count} image(s) for '{display_name}' → +{points} points! 🎉<br>"
          f"<strong>Suggested profile completed — removed from list!</strong>", "success")

    if feature_ok_count:
        flash(f"Indexed {feature_ok_count}/{saved_count} new image(s) into incremental feature DB.", "info")
    if feature_fail_count:
        flash(f"Warning: failed to index {feature_fail_count} image(s). Check server logs.", "warning")

    final_total = existing_count + saved_count
    if final_total >= MAX_IMAGES_PER_PERSON:
        flash(f"Maximum {MAX_IMAGES_PER_PERSON} images now reached for '{display_name}'.", "info")

    return redirect(url_for('contribute'))

@app.route('/search', methods=['POST'])
@login_required
def search():
    """
    Handle face search upload:
    - Save uploaded image temporarily
    - Run reverse image search against FAISS index
    - Show results
    - If super-user + poor/no match → allow adding as new suggestion to missminimized.csv
    """
    # ── 1. Basic input validation ───────────────────────────────────────
    if 'image' not in request.files:
        flash("No image was uploaded.", "error")
        return redirect(url_for('upload_form'))

    file = request.files['image']
    if not file or not file.filename:
        flash("No file selected.", "error")
        return redirect(url_for('upload_form'))

    # Accept common image formats
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    if not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
        flash("Only JPG, JPEG, PNG, or WEBP files are allowed.", "error")
        return redirect(url_for('upload_form'))

    # ── 2. Save uploaded file temporarily ────────────────────────────────
    from werkzeug.utils import secure_filename  # Secure the filename
    query_filename = secure_filename(file.filename)
    query_path = os.path.join(uploads_dir, query_filename)

    try:
        file.save(query_path)
    except Exception as e:
        flash(f"Failed to save uploaded image: {str(e)}", "error")
        return redirect(url_for('upload_form'))

    # Create base64 preview for the template
    try:
        with open(query_path, "rb") as f:
            img_base64 = f"data:image/jpeg;base64,{base64.b64encode(f.read()).decode('utf-8')}"
    except Exception as e:
        flash("Failed to generate image preview.", "warning")
        img_base64 = ""

    # ── 3. Run the actual face search ────────────────────────────────────
    try:
        results, detection_method = main(query_path, top_k=30)
    except Exception as e:
        flash(f"Search failed: {str(e)}", "error")
        try:
            os.remove(query_path)
        except:
            pass
        return render_template('error.html', error=str(e)), 500

    # ── 4. Determine if we have a "strong" match ─────────────────────────
    strong_match_found = False
    min_distance = float('inf')
    closest_name = "Unknown"
    should_deduct_point = False

    if results:
        distances = [r['distance'] for r in results]
        min_distance = min(distances)
        best_idx = distances.index(min_distance)
        best_result = results[best_idx]

        if min_distance < 1.0:  # ← adjust threshold if needed (e.g. 0.95, 1.1, ...)
            strong_match_found = True

            # Try to get human-readable name from db.txt
            filename = os.path.basename(best_result['image_path'])
            img_id = filename.split('_')[0]

            for line in db_lines:
                if line.strip().startswith(img_id + ','):
                    parts = line.strip().split(',', 1)
                    if len(parts) == 2:
                        closest_name = parts[1].strip()
                        break

            # Decide whether to deduct point
            if 'common' in closest_name.lower():
                flash(f"Free search — '{closest_name}' is marked as common → no points deducted", "success")
            else:
                should_deduct_point = True

    # ── 5. Deduct point if needed ───────────────────────────────────────
    if should_deduct_point:
        current_points = session.get('points', 0)
        if current_points < 1:
            flash("Not enough points to view this result!", "error")
            try:
                os.remove(query_path)
            except:
                pass
            return redirect(url_for('upload_form'))

        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("UPDATE users SET points = points - 1 WHERE user_id = ?", (session['user_id'],))
            conn.commit()
            conn.close()

            session['points'] = current_points - 1
            flash(f"Strong match found ('{closest_name}', dist {min_distance:.3f}) → -1 point", "info")
        except Exception as e:
            flash(f"Failed to deduct point: {str(e)}", "error")

    elif strong_match_found:
        flash(f"Strong match found (dist {min_distance:.3f}) — no deduction needed", "success")
    else:
        # FIXED: Correct f-string formatting
        distance_str = f"{min_distance:.3f}" if results else "N/A"
        flash(f"No strong match (best distance: {distance_str}) → no points deducted", "info")

    # ── 6. Enrich results with names & links ─────────────────────────────
    # Build lookup maps once
    db_map = {}
    for line in db_lines:
        stripped = line.strip()
        if ',' in stripped:
            parts = stripped.split(',', 1)
            db_map[parts[0].strip()] = parts[1].strip()
    blank_map = load_blank_map()

    saved_map = {}
    for line in saved_lines:
        stripped = line.strip()
        if ',' in stripped:
            parts = stripped.split(',')
            saved_map[parts[0].strip()] = parts[1].strip()

    enriched_results = []
    for r in results or []:  # Safe iteration even if no results
        filename = r['image_path'].split('.')[0]  # Get just the filename
        parts = filename.split('_')
        img_id = parts[0]
        entry = {
            'distance': r['distance'], 
            'image_path': filename,
            'index': r.get('index', 0)
        }
        if re.match(r'^\d{6}$', img_id) and len(parts) > 1:
            entry['name'] = db_map.get(img_id, blank_map.get(img_id, 'Unknown'))
            entry['link'] = ''
            entry['id'] = filename
        else:
            try:
                if '00001' in filename:
                    entry['name'] = filename.rsplit("_", 1)[0]
                    entry['link'] = ''
                    entry['id'] = ''
                else:
                    entry['name'] = saved_map.get(filename, '')
                    entry['link'] = filename  # Assuming the link is just the ID for saved entries
                    entry['id'] = filename
            except:
                entry['name'] = filename
                entry['link'] = ''
                entry['id'] = ''
        enriched_results.append(entry)

    # ── 7. Render results ────────────────────────────────────────────────
    is_super = session.get('user_email') == SUPER_USER_EMAIL

    return render_template(
        'results.html',
        img_base64=img_base64,
        detection_method=detection_method,
        results=enriched_results,
        min_distance=min_distance if results else None,
        strong_match=strong_match_found,
        query_filename=query_filename,
        is_super_user=is_super,           # ← boolean, not the function
    )

@app.route('/add_new_person', methods=['POST'])
@login_required
def add_new_person():
    if not is_super_user():
        flash("Only super user can add new identities.", "error")
        return redirect(url_for('index'))

    new_name = request.form.get('new_person_name', '').strip()
    profile_link = request.form.get('profile_link', '').strip()

    if not new_name:
        flash("Person name is required.", "error")
        return redirect(url_for('search'))

    # ── Step 1: Read all lines ────────────────────────────────────────
    try:
        with open(BLANK_TXT, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        flash("Suggestions file not found.", "error")
        return redirect(url_for('index'))

    # ── Step 2: Find the first blank name line ─────────────────────────
    selected_id = None
    selected_line_index = -1

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            id_part, rest = line.split(',', 1)
            id_part = id_part.strip()
            name_part = rest.strip()
            # We consider it "blank" if there's nothing meaningful after the comma
            if not name_part or name_part == '':
                selected_id = id_part
                selected_line_index = i
                break
        except ValueError:
            # malformed line - skip
            continue

    if selected_id is None:
        flash("No more blank slots available in missminimized.csv", "error")
        return redirect(url_for('search'))

    str_id = selected_id  # e.g. "161308"

    # ── Step 3: Save image ─────────────────────────────────────────────
    query_filename = request.form.get('query_filename', '')
    if not query_filename:
        flash("Original image information lost.", "error")
        return redirect(url_for('index'))

    src_path = os.path.join(uploads_dir, query_filename)
    if not os.path.exists(src_path):
        flash("Original query image no longer available.", "error")
        return redirect(url_for('index'))

    contrib_folder = os.path.join(contributions_dir, str_id)
    os.makedirs(contrib_folder, exist_ok=True)

    ext = os.path.splitext(query_filename)[1] or '.jpg'
    dest_filename = f"{str_id}_from_search{ext}"
    dest_path = os.path.join(contrib_folder, dest_filename)

    import shutil
    shutil.copy2(src_path, dest_path)

    # Clean up temp file
    try:
        os.remove(src_path)
    except:
        pass

    # ── Step 4: Update the line in missminimized.csv ───────────────────
    # Replace the blank name with the real one
    new_content = f"{str_id},{new_name}"
    if profile_link:
        new_content += f"/{profile_link}"

    # Rewrite the file with the updated line
    lines[selected_line_index] = new_content + '\n'

    try:
        with open(BLANK_TXT, 'w', encoding='utf-8') as f:
            f.writelines(lines)
    except Exception as e:
        flash(f"Failed to update suggestions file: {str(e)}", "error")
        return redirect(url_for('index'))

    # ── Step 5: Award points ───────────────────────────────────────────
    feature_ok, feature_msg = auto_update_incremental_feature(dest_path, os.path.basename(dest_path))
    award_points(session['user_id'], 1, person_name=new_name, image_count=1)

    # ── Success message ────────────────────────────────────────────────
    flash(
        f"Success! Used blank slot → **{new_name}** (ID **{str_id}**) added to missminimized.csv → +5 points!<br>"
        f"Image saved → <code>contributions/{str_id}/{dest_filename}</code><br>"
        f"The entry is now marked and will appear in contribution queue.",
        "success"
    )

    if feature_ok:
        flash("Indexed new identity into incremental feature DB.", "info")
    else:
        flash(f"Warning: feature indexing failed: {feature_msg}", "warning")

    return redirect(url_for('contribute'))  # or 'search' or 'index'

# ========================
# Jinja
# ========================
app.jinja_env.filters['basename'] = os.path.basename

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
