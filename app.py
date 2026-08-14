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
# CONFIGURATION
# ============================================================

MODEL_PATH = "gender_classification_model.h5"

# Model input size
IMG_WIDTH = 150
IMG_HEIGHT = 200

# Processing resolution
PROCESS_WIDTH = 480
PROCESS_HEIGHT = 360

# ============================================================
# LOAD GENDER MODEL
# ============================================================

model = None

try:
    model = load_model(MODEL_PATH, compile=False)
    print("==========================================")
    print("MODEL LOADED SUCCESSFULLY")
    print("==========================================")
except Exception as e:
    print("==========================================")
    print("MODEL LOAD ERROR")
    print(str(e))
    print("==========================================")

# ============================================================
# LOAD HAAR FACE DETECTOR
# ============================================================

cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    print("FACE CASCADE LOAD ERROR")
else:
    print("FACE CASCADE LOADED SUCCESSFULLY")
    print("Cascade path:", cascade_path)

# ============================================================
# LABELS
# ============================================================

labels = ["Female", "Male"]

# ============================================================
# FACE DETECTION
# ============================================================

def detect_faces(gray):
    """
    Detect faces using OpenCV Haar Cascade.
    """

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40)
    )

    return faces


# ============================================================
# PROCESS FRAME
# ============================================================

def process_frame(frame):

    predictions = []

    if frame is None:
        return frame, predictions

    if model is None:
        print("MODEL IS NOT LOADED")
        return frame, predictions

    # --------------------------------------------------------
    # Resize frame for faster processing
    # --------------------------------------------------------

    original_height, original_width = frame.shape[:2]

    if original_width > PROCESS_WIDTH:

        scale = PROCESS_WIDTH / original_width

        new_width = PROCESS_WIDTH
        new_height = int(original_height * scale)

        frame = cv2.resize(
            frame,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA
        )

    # --------------------------------------------------------
    # Convert to grayscale
    # --------------------------------------------------------

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Improve contrast slightly
    gray = cv2.equalizeHist(gray)

    # --------------------------------------------------------
    # Detect faces
    # --------------------------------------------------------

    faces = detect_faces(gray)

    print(
        f"Frame: {frame.shape[1]}x{frame.shape[0]} | "
        f"Faces detected: {len(faces)}"
    )

    # --------------------------------------------------------
    # Process each detected face
    # --------------------------------------------------------

    for (x, y, w, h) in faces:

        try:

            # Extract face
            face = frame[y:y + h, x:x + w]

            if face is None or face.size == 0:
                continue

            # ------------------------------------------------
            # Resize face to model input
            # ------------------------------------------------

            face_resized = cv2.resize(
                face,
                (IMG_WIDTH, IMG_HEIGHT),
                interpolation=cv2.INTER_AREA
            )

            # ------------------------------------------------
            # Convert image to array
            # ------------------------------------------------

            face_array = img_to_array(face_resized)

            # Normalize exactly like training
            face_array = face_array / 255.0

            # Add batch dimension
            face_array = np.expand_dims(face_array, axis=0)

            # ------------------------------------------------
            # Gender prediction
            # ------------------------------------------------

            prediction = model.predict(
                face_array,
                verbose=0
            )[0][0]

            prediction = float(prediction)

            # ------------------------------------------------
            # Calculate confidence
            # ------------------------------------------------

            male_confidence = prediction
            female_confidence = 1.0 - prediction

            if male_confidence >= 0.5:

                gender = "Male"
                confidence = male_confidence

                # Blue
                color = (255, 0, 0)

            else:

                gender = "Female"
                confidence = female_confidence

                # Magenta
                color = (255, 0, 255)

            # ------------------------------------------------
            # Draw bounding box
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                color,
                2
            )

            # ------------------------------------------------
            # Label
            # ------------------------------------------------

            label = f"{gender} - {confidence:.1%}"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2

            (label_width, label_height), baseline = cv2.getTextSize(
                label,
                font,
                font_scale,
                thickness
            )

            # Make sure label doesn't go above image
            label_y = max(y, label_height + 10)

            # Background
            cv2.rectangle(
                frame,
                (x, label_y - label_height - 10),
                (x + label_width + 10, label_y),
                color,
                -1
            )

            # Text
            cv2.putText(
                frame,
                label,
                (x + 5, label_y - 5),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA
            )

            # ------------------------------------------------
            # Store prediction
            # ------------------------------------------------

            predictions.append({
                "gender": gender,
                "confidence": round(confidence, 4),
                "male_confidence": round(male_confidence, 4),
                "female_confidence": round(female_confidence, 4),
                "bbox": [
                    int(x),
                    int(y),
                    int(w),
                    int(h)
                ]
            })

            print(
                f"Prediction: {gender} | "
                f"Confidence: {confidence:.2%}"
            )

        except Exception as e:

            print(
                "FACE PROCESSING ERROR:",
                str(e)
            )

    return frame, predictions


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def index():

    return render_template("index.html")


# ============================================================
# PROCESS FRAME API
# ============================================================

@app.route("/process_frame", methods=["POST"])
def process_frame_route():

    start_time = time.time()

    try:

        # ----------------------------------------------------
        # Get JSON
        # ----------------------------------------------------

        data = request.get_json(silent=True)

        if not data:

            return jsonify({
                "success": False,
                "error": "No JSON data received"
            }), 400

        if "image" not in data:

            return jsonify({
                "success": False,
                "error": "No image received"
            }), 400

        image_data = data["image"]

        # ----------------------------------------------------
        # Remove Data URL prefix
        # ----------------------------------------------------

        if "," in image_data:

            image_data = image_data.split(",", 1)[1]

        # ----------------------------------------------------
        # Decode Base64
        # ----------------------------------------------------

        try:

            image_bytes = base64.b64decode(
                image_data,
                validate=True
            )

        except Exception:

            return jsonify({
                "success": False,
                "error": "Invalid Base64 image"
            }), 400

        # ----------------------------------------------------
        # Convert bytes to NumPy
        # ----------------------------------------------------

        nparr = np.frombuffer(
            image_bytes,
            np.uint8
        )

        # ----------------------------------------------------
        # Decode JPEG
        # ----------------------------------------------------

        frame = cv2.imdecode(
            nparr,
            cv2.IMREAD_COLOR
        )

        if frame is None:

            return jsonify({
                "success": False,
                "error": "OpenCV could not decode image"
            }), 400

        print(
            f"Received frame: "
            f"{frame.shape[1]}x{frame.shape[0]}"
        )

        # ----------------------------------------------------
        # Process frame
        # ----------------------------------------------------

        processed_frame, predictions = process_frame(frame)

        # ----------------------------------------------------
        # Encode processed frame
        # ----------------------------------------------------

        success, buffer = cv2.imencode(
            ".jpg",
            processed_frame,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                65
            ]
        )

        if not success:

            return jsonify({
                "success": False,
                "error": "JPEG encoding failed"
            }), 500

        # ----------------------------------------------------
        # Convert JPEG to Base64
        # ----------------------------------------------------

        processed_image = base64.b64encode(
            buffer
        ).decode("utf-8")

        processing_time = time.time() - start_time

        print(
            f"Processing time: "
            f"{processing_time:.3f}s"
        )

        # ----------------------------------------------------
        # Return result
        # ----------------------------------------------------

        return jsonify({

            "success": True,

            "image":
                "data:image/jpeg;base64,"
                + processed_image,

            "predictions": predictions,

            "processing_time":
                round(processing_time, 3)

        })

    except Exception as e:

        print(
            "PROCESS FRAME ERROR:",
            str(e)
        )

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health")
def health():

    return jsonify({

        "status": "ok",

        "model_loaded":
            model is not None,

        "face_detector_loaded":
            not face_cascade.empty(),

        "model_path":
            MODEL_PATH,

        "model_input":
            f"{IMG_HEIGHT}x{IMG_WIDTH}x3"

    })


# ============================================================
# TEST FACE DETECTOR
# ============================================================

@app.route("/test")
def test():

    return jsonify({

        "message":
            "Gender detection Flask API is running",

        "model_loaded":
            model is not None,

        "face_detector_loaded":
            not face_cascade.empty()

    })


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            7860
        )
    )

    print("==========================================")
    print("Starting Gender Detection Flask Server")
    print("Port:", port)
    print("==========================================")

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        threaded=True
    )