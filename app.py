from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from keras.models import load_model
from keras.utils import img_to_array
import base64
import os
import time

app = Flask(__name__)

# ============================================================
# Configuration
# ============================================================

MODEL_PATH = "gender_classification_model.h5"
IMG_WIDTH = 150
IMG_HEIGHT = 200
PROCESS_WIDTH = 320  # Reduce image size for faster processing
PROCESS_HEIGHT = 240

# ============================================================
# Load Model
# ============================================================

try:
    model = load_model(MODEL_PATH, compile=False)
    print("✅ Gender classification model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# ============================================================
# Load Face Detection Model
# ============================================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    print("❌ Error loading Haar Cascade")
else:
    print("✅ Face detection model loaded successfully")

# ============================================================
# Labels
# ============================================================

labels = ["Female", "Male"]

# ============================================================
# Optimized Face Detection with Caching
# ============================================================

# Cache for face detection results
face_cache = {
    'last_detection_time': 0,
    'last_faces': [],
    'cache_duration': 0.3  # Cache face positions for 300ms
}

def detect_faces_optimized(gray):
    """Detect faces with caching for better performance"""
    current_time = time.time()
    
    # Return cached faces if within cache duration
    if current_time - face_cache['last_detection_time'] < face_cache['cache_duration']:
        return face_cache['last_faces']
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    # Update cache
    face_cache['last_faces'] = faces
    face_cache['last_detection_time'] = current_time
    
    return faces

def process_frame(frame):
    """Process frame with optimized face detection"""
    predictions = []

    if frame is None or model is None:
        return frame, predictions

    # Resize frame for faster processing
    height, width = frame.shape[:2]
    
    # Only resize if frame is larger than processing size
    if width > PROCESS_WIDTH:
        scale = PROCESS_WIDTH / width
        new_height = int(height * scale)
        frame = cv2.resize(frame, (PROCESS_WIDTH, new_height))

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces with caching
    faces = detect_faces_optimized(gray)

    # Process each face
    for (x, y, w, h) in faces:
        try:
            # Extract face
            face = frame[y:y + h, x:x + w]
            
            if face.size == 0:
                continue

            # Resize to model input size
            face_resized = cv2.resize(face, (IMG_WIDTH, IMG_HEIGHT))
            
            # Preprocess
            face_array = img_to_array(face_resized) / 255.0
            face_array = np.expand_dims(face_array, axis=0)
            
            # Predict (batch prediction if multiple faces)
            prediction = model.predict(face_array, verbose=0)[0][0]
            prediction = float(prediction)
            
            # Calculate confidences
            male_confidence = prediction
            female_confidence = 1.0 - prediction
            
            # Determine gender
            if male_confidence >= female_confidence:
                gender = "Male"
                confidence = male_confidence
                color = (255, 0, 0)  # Blue for Male
            else:
                gender = "Female"
                confidence = female_confidence
                color = (255, 0, 255)  # Magenta for Female
            
            # Draw simple bounding box (no complex corners for speed)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw label with confidence
            label = f"{gender}: {confidence:.1%}"
            
            # Simple label background
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x, y - label_h - 10), (x + label_w, y), color, -1)
            cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Store prediction
            predictions.append({
                "gender": gender,
                "confidence": confidence,
                "male_confidence": male_confidence,
                "female_confidence": female_confidence,
                "bbox": [int(x), int(y), int(w), int(h)]
            })
            
        except Exception as e:
            print(f"Error processing face: {e}")
            continue

    return frame, predictions

# ============================================================
# Routes
# ============================================================

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/process_frame", methods=["POST"])
def process_frame_route():
    try:
        data = request.get_json()
        
        if not data or "image" not in data:
            return jsonify({"success": False, "error": "No image"}), 400

        # Get base64 image
        image_data = data["image"]
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]

        # Decode image
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({"success": False, "error": "Decode failed"}), 400

        # Process frame
        processed_frame, predictions = process_frame(frame)
        
        # Encode with lower quality for faster transfer
        success, buffer = cv2.imencode(".jpg", processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
        
        if not success:
            return jsonify({"success": False, "error": "Encoding failed"}), 500

        # Convert to Base64
        processed_image = base64.b64encode(buffer).decode("utf-8")
        
        return jsonify({
            "success": True,
            "image": f"data:image/jpeg;base64,{processed_image}",
            "predictions": predictions
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)