import cv2
import numpy as np
import faiss
import logging
import os
from typing import List, Dict, Tuple
from scipy.spatial.transform import Rotation
import insightface
from flask import Flask, request, jsonify

app = Flask(__name__)

# Configuration: Dynamic paths for local/Codespace vs Render
if os.getenv('RENDER') == 'true':
    # Render environment
    base_dir = '/app'
else:
    # Local or Codespace environment
    base_dir = os.path.dirname(os.path.abspath(__file__))  # /mnt/e/workout/practice/python/urfavid or Codespace workspace

features_file = os.path.join(base_dir, "data", "features3_fusion_merged.npy")
report_file = os.path.join(base_dir, "data", "search_report.txt")
debug_dir = os.path.join(base_dir, "data", "debug_images")
uploads_dir = os.path.join(base_dir, "data", "uploads")
os.makedirs(debug_dir, exist_ok=True)
os.makedirs(uploads_dir, exist_ok=True)

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(base_dir, "data", "search_log.txt")),
        logging.StreamHandler()
    ]
)

# Initialize InsightFace
try:
    face_analysis = insightface.app.FaceAnalysis(det_name='retinaface_r50_v1', rec_name='arcface_r100_v1')
    face_analysis.prepare(ctx_id=-1, det_size=(320, 320))  # Reduced for memory
    logging.info("InsightFace initialized successfully")
except Exception as e:
    logging.error(f"InsightFace initialization failed: {str(e)}")
    raise

# Cache for FAISS index
global_index = None
global_image_paths = None

def preprocess_image(image: np.ndarray, contrast_factor: float = 1.0, resolution_factor: float = 1.0) -> np.ndarray:
    try:
        if image is None or len(image.shape) != 3:
            raise ValueError("Invalid image: Image is None or not in correct format")
        max_width = int(320 * resolution_factor)
        height, width = image.shape[:2]
        if width > max_width:
            scale = max_width / width
            image = cv2.resize(image, (max_width, int(height * scale)), interpolation=cv2.INTER_AREA)
        if contrast_factor != 1.0:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=contrast_factor, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return image
    except Exception as e:
        logging.warning(f"Image preprocessing failed: {str(e)}")
        raise

def estimate_pose(points: np.ndarray) -> Tuple[float, float, float]:
    try:
        if not isinstance(points, np.ndarray) or points.shape != (68, 2):
            raise ValueError(f"Invalid points shape: {points.shape}")
        left_eye_center = np.mean(points[36:42], axis=0)
        right_eye_center = np.mean(points[42:48], axis=0)
        nose_tip = points[30]
        dY = right_eye_center[1] - left_eye_center[1]
        dX = right_eye_center[0] - right_eye_center[0] + 1e-6
        roll = np.degrees(np.arctan2(dY, dX))
        eye_midpoint = (left_eye_center + right_eye_center) / 2
        pitch = np.degrees(np.arctan2(nose_tip[1] - eye_midpoint[1], 100.0))
        face_center = np.mean(points[0:17], axis=0)
        yaw = np.degrees(np.arctan2(nose_tip[0] - face_center[0], 100.0))
        return roll, pitch, yaw
    except Exception as e:
        logging.warning(f"Pose estimation failed: {str(e)}")
        return 0.0, 0.0, 0.0

def compute_curvature(points: np.ndarray, indices: range) -> float:
    try:
        if not isinstance(points, np.ndarray) or points.shape != (68, 2):
            raise ValueError(f"Invalid points shape: {points.shape}")
        selected_points = points[list(indices)]
        x, y = selected_points[:, 0], selected_points[:, 1]
        coeffs = np.polyfit(x, y, 2)
        return abs(coeffs[0])
    except Exception as e:
        logging.warning(f"Curvature computation failed: {str(e)}")
        return 0.0

def extract_facial_features(image: np.ndarray, image_path: str) -> Dict:
    try:
        processed_image = preprocess_image(image)
        logging.info(f"Preprocessed image: {image_path}")
        faces = face_analysis.get(processed_image)
        if not faces:
            logging.warning(f"No faces detected in {image_path}")
            return {'features': None, 'image_path': image_path, 'status': 'failed',
                    'reason': 'No faces detected', 'detection_method': 'insightface', 'confidence': 0.0}
        areas = [(f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]) for f in faces]
        max_idx = np.argmax(areas)
        selected_face = faces[max_idx]
        embedding = selected_face.embedding / np.linalg.norm(selected_face.embedding)
        confidence = selected_face.det_score
        additional_features = np.zeros(7, dtype=np.float32)
        landmarks = selected_face.landmark
        if landmarks is not None and isinstance(landmarks, np.ndarray) and landmarks.shape == (68, 2):
            roll, pitch, yaw = estimate_pose(landmarks)
            jaw_curvature = compute_curvature(landmarks, range(0, 17))
            left_eyebrow_curvature = compute_curvature(landmarks, range(17, 22))
            right_eyebrow_curvature = compute_curvature(landmarks, range(22, 27))
            lip_curvature = compute_curvature(landmarks, range(48, 55))
            additional_features = np.array([roll / 180.0, pitch / 180.0, yaw / 180.0,
                                           jaw_curvature, left_eyebrow_curvature,
                                           right_eyebrow_curvature, lip_curvature], dtype=np.float32)
        else:
            logging.warning(f"Landmarks missing or invalid for {image_path}, using default features")
            confidence *= 0.8
        features = np.concatenate([embedding, additional_features])
        logging.info(f"Extracted features for {image_path}, dimension: {len(features)}, confidence: {confidence:.4f}")
        debug_img = processed_image.copy()
        bbox = selected_face.bbox.astype(int)
        cv2.rectangle(debug_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        if landmarks is not None and isinstance(landmarks, np.ndarray) and landmarks.shape == (68, 2):
            for x, y in landmarks:
                cv2.circle(debug_img, (int(x), int(y)), 2, (0, 0, 255), -1)
        debug_path = os.path.join(debug_dir, f"{os.path.basename(image_path)}_insightface_debug.jpg")
        cv2.imwrite(debug_path, debug_img)
        logging.info(f"Saved debug image: {debug_path}")
        return {'features': features, 'image_path': image_path, 'status': 'success',
                'reason': None, 'detection_method': 'insightface', 'confidence': confidence}
    except Exception as e:
        logging.error(f"Feature extraction failed for {image_path}: {str(e)}")
        debug_path = os.path.join(debug_dir, f"{os.path.basename(image_path)}_failed.jpg")
        cv2.imwrite(debug_path, image)
        return {'features': None, 'image_path': image_path, 'status': 'failed',
                'reason': f'Exception: {str(e)}', 'detection_method': 'insightface', 'confidence': 0.0}

def load_existing_features(features_file: str) -> Tuple[List[np.ndarray], List[str]]:
    try:
        if not os.path.exists(features_file):
            raise FileNotFoundError(f"Features file not found: {features_file}")
        data = np.load(features_file, allow_pickle=True).item()
        feature_vectors = data.get('feature_vectors', [])
        image_paths = data.get('image_paths', [])
        if not feature_vectors:
            raise ValueError("No feature vectors found in the database")
        dimension = len(feature_vectors[0])
        for i, vec in enumerate(feature_vectors):
            if len(vec) != dimension:
                raise ValueError(f"Inconsistent feature vector dimension at index {i}: expected {dimension}, got {len(vec)}")
        logging.info(f"Loaded {len(feature_vectors)} feature vectors with dimension {dimension} from {features_file}")
        return feature_vectors, image_paths
    except Exception as e:
        logging.error(f"Failed to load existing features from {features_file}: {str(e)}")
        raise

def build_feature_index(feature_vectors: List[np.ndarray]) -> faiss.Index:
    try:
        if not feature_vectors:
            raise ValueError("Feature vector list is empty")
        dimension = len(feature_vectors[0])
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(feature_vectors).astype(np.float32))
        logging.info(f"Built FAISS index with {len(feature_vectors)} vectors of dimension {dimension}")
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
        logging.info(f"Performed search, found {len(indices[0])} matches")
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

def load_and_cache_index():
    global global_index, global_image_paths
    if global_index is None:
        feature_vectors, image_paths = load_existing_features(features_file)
        global_index = build_feature_index(feature_vectors)
        global_image_paths = image_paths
    return global_index, global_image_paths

def main(top_k: int = 30):
    try:
        index, image_paths = load_and_cache_index()
        results, query_detection_method = reverse_image_search(request.form.get('query_image_path'), index, image_paths, top_k=top_k)
        report = []
        report.append(f"Query Image: {os.path.basename(request.form.get('query_image_path'))}")
        report.append(f"Status: Success")
        report.append(f"Detection Method: {query_detection_method}")
        report.append(f"Top {top_k} Matches:")
        for result in results:
            report.append(f"  Match: {os.path.basename(result['image_path'])}, Distance: {result['distance']:.4f}")
        with open(report_file, 'w') as f:
            f.write("\n".join(report))
        logging.info(f"Search report saved to {report_file}")
        return results, query_detection_method
    except Exception as e:
        logging.error(f"Main execution failed: {str(e)}")
        error_report = [f"Query Image: {os.path.basename(request.form.get('query_image_path', 'unknown'))}"]
        error_report.append(f"Status: Failed")
        error_report.append(f"Reason: {str(e)}")
        with open(report_file, 'w') as f:
            f.write("\n".join(error_report))
        raise

@app.route('/')
def index():
    return "Flask server is running. Use POST /search to upload an image."

@app.route('/search', methods=['POST'])
def search():
    try:
        if 'image' not in request.files:
            return jsonify({'status': 'failed', 'error': 'No image uploaded'}), 400
        file = request.files['image']
        query_image_path = os.path.join(uploads_dir, file.filename)
        file.save(query_image_path)
        request.form = {'query_image_path': query_image_path}  # Simulate form data
        results, detection_method = main(top_k=30)
        return jsonify({
            'status': 'success',
            'detection_method': detection_method,
            'results': [
                {'image_path': os.path.basename(r['image_path']), 'distance': r['distance'], 'index': r['index']}
                for r in results
            ]
        })
    except Exception as e:
        return jsonify({'status': 'failed', 'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)