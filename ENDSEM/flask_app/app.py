# flask_app/app.py
# Final complete version with analysis, labeling, and support for the requested layout.

import os
import uuid
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
import traceback
from skimage.morphology import skeletonize # For vein analysis
import math # For distance calculation

# --- Configuration ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress TF info logs
MODEL_PATH = MODEL_PATH = r'D:\collegeStuff\SEM6\COMPUTER_VISION\PROJECt\CV\ENDSEM\models\unet_multi'   # Raw string for Windows path
INPUT_MODEL_DIMENSION = 512
DEFAULT_PIXELS_PER_INCH = 96

# --- Flask App Setup ---
app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/' # Change for production
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
MASKS_FOLDER = os.path.join(BASE_DIR, 'static', 'masks')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff'}

# Create static subdirectories if they don't exist
try:
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(MASKS_FOLDER, exist_ok=True)
    print(f"Static folders checked/created: {UPLOAD_FOLDER}, {MASKS_FOLDER}")
except OSError as e:
    print(f"CRITICAL ERROR creating static folders: {e}")
    # exit(1) # Optionally exit if folders are essential

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MASKS_FOLDER'] = MASKS_FOLDER

# --- Global model variables ---
unet_model = None
inference_func = None

# --- Load TensorFlow Model Function ---
def load_tf_model():
    """Loads the TensorFlow SavedModel and sets global variables."""
    global unet_model, inference_func
    print("-----------------------------------------")
    print("Attempting to load TensorFlow model...")
    try:
        if not os.path.isdir(MODEL_PATH):
            raise FileNotFoundError(f"Model directory not found or is not a directory: {MODEL_PATH}")

        print(f"Calling tf.saved_model.load('{MODEL_PATH}')...")
        unet_model = tf.saved_model.load(MODEL_PATH)
        inference_func = unet_model.signatures["serving_default"]
        print("TensorFlow model loaded successfully.")

        gpus = tf.config.list_physical_devices('GPU')
        print(f"GPUs detected by TensorFlow after load: {gpus if gpus else 'None - Running on CPU'}")
        print("-----------------------------------------")
        return True

    except Exception as e:
        print(f"FATAL ERROR: Failed to load TensorFlow model from {MODEL_PATH}")
        print(f"Error details: {e}\nFull traceback:\n{traceback.format_exc()}")
        print("-----------------------------------------")
        unet_model = None
        inference_func = None
        return False

# --- Helper Functions ---
def allowed_file(filename):
    """Checks if the file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path, target_size):
    """Loads and preprocesses an image for the UNet model."""
    print(f"Preprocessing image: {image_path}")
    try:
        image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            print(f"Error: cv2.imread failed for {image_path}")
            return None
        image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image_clahe = clahe.apply(image_gray)
        image_resized = cv2.resize(image_clahe, (target_size, target_size), interpolation=cv2.INTER_AREA)
        image_normalized = image_resized.astype(dtype=np.float32) / 127.5 - 1
        image_expanded = np.expand_dims(np.expand_dims(image_normalized, axis=0), axis=3)
        image_tensor = tf.convert_to_tensor(image_expanded, dtype=tf.float32)
        print("Preprocessing successful.")
        return image_tensor
    except Exception as e:
        print(f"Error during preprocessing for {image_path}: {e}\n{traceback.format_exc()}")
        return None

def generate_visual_mask(predictions, output_mask_path):
    """Generates and saves a Black & White visual mask."""
    print(f"Generating B&W visual mask: {output_mask_path}")
    try:
        mask_squeezed = np.squeeze(predictions)
        prediction_mask = np.argmax(mask_squeezed, axis=2).astype(np.uint8)
        output_visual_gray = np.zeros((prediction_mask.shape[0], prediction_mask.shape[1]), dtype=np.uint8)
        # Class 2 = Veins = White
        output_visual_gray[prediction_mask == 2] = 255
        if cv2.imwrite(output_mask_path, output_visual_gray):
            print(f"B&W Visual mask saved successfully: {output_mask_path}")
            return True
        else:
            print(f"Error: cv2.imwrite failed for B&W mask: {output_mask_path}")
            return False
    except Exception as e:
        print(f"Error generating/saving B&W visual mask: {e}\n{traceback.format_exc()}")
        return False

def analyze_mask_for_sites(mask_filepath, scale_pixels_per_inch):
    """Analyzes skeleton, finds potential sites, adds labels."""
    print(f"Analyzing mask: {mask_filepath} with scale: {scale_pixels_per_inch} px/inch")
    analysis_results = []
    annotated_filepath = None
    try:
        mask = cv2.imread(mask_filepath, cv2.IMREAD_GRAYSCALE)
        if mask is None: return None, [], "Analysis Error: Could not read mask file."

        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        skeleton_img = skeletonize(binary_mask / 255).astype(np.uint8) * 255
        if np.sum(skeleton_img) == 0: return None, [], "Analysis: No structures found."
        print("Skeletonization complete.")

        # Create color image: White skeleton on black background for drawing
        annotated_image = cv2.cvtColor(skeleton_img, cv2.COLOR_GRAY2BGR)
        annotated_image[skeleton_img == 255] = [255, 255, 255]

        # --- Analysis Parameters ---
        min_length_pixels = 0.25 * scale_pixels_per_inch
        min_straightness_ratio = 0.85 # Adjust if needed
        min_contour_points = 5      # Ignore tiny fragments

        # --- Font settings for labels ---
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_color = (0, 255, 255) # Yellow (BGR) - contrast on green/white/black
        font_thickness = 1
        label_offset_x = 5
        label_offset_y = -5

        contours, _ = cv2.findContours(skeleton_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        print(f"Found {len(contours)} raw contours.")

        suitable_site_count = 0
        rejected_short_count = 0
        rejected_straight_count = 0

        for i, contour in enumerate(contours):
            reason = []; passes_checks = True
            if len(contour) < min_contour_points: continue

            length_px = cv2.arcLength(contour, closed=False)
            length_inches = length_px / scale_pixels_per_inch
            if length_px >= min_length_pixels: reason.append(f"L({length_inches:.2f}\")≥0.25\"")
            else: passes_checks = False; rejected_short_count += 1; continue

            if len(contour) >= min_contour_points:
                try:
                    rect = cv2.minAreaRect(contour); box = cv2.boxPoints(rect); box = np.intp(box)
                    diag1 = math.dist(box[0], box[2]); diag2 = math.dist(box[1], box[3]); max_diag = max(diag1, diag2)
                    straightness_ratio = length_px / max_diag if max_diag > 0 else 0
                    if straightness_ratio >= min_straightness_ratio: reason.append(f"S({straightness_ratio:.2f})≥{min_straightness_ratio:.2f}")
                    else: passes_checks = False; rejected_straight_count += 1; continue
                except Exception: passes_checks = False; continue # Handle potential cv2 errors
            else: passes_checks = False; continue

            reason.append("Junctions(N/A)") # Placeholder

            if passes_checks:
                suitable_site_count += 1
                cv2.drawContours(annotated_image, [contour], -1, (0, 255, 0), 2) # Draw GREEN

                M = cv2.moments(contour)
                cX = int(M["m10"] / M["m00"]) if M["m00"] != 0 else int(np.mean(contour[:,0,0]))
                cY = int(M["m01"] / M["m00"]) if M["m00"] != 0 else int(np.mean(contour[:,0,1]))

                label_text = str(suitable_site_count) # Label with its number
                text_origin = (cX + label_offset_x, cY + label_offset_y)
                # Add text label near the center point
                cv2.putText(annotated_image, label_text, text_origin, font, font_scale, font_color, font_thickness, cv2.LINE_AA)

                analysis_results.append({ 'id': suitable_site_count, 'location': (cX, cY),
                                          'length_in': length_inches, 'straightness': straightness_ratio,
                                          'reason': " & ".join(reason) })

        # --- Final Message & Save ---
        if suitable_site_count > 0: analysis_message = f"Found {suitable_site_count} potential segment(s) meeting basic criteria (labeled green)."
        else: analysis_message = f"No segments met basic criteria (L≥0.25\"@{scale_pixels_per_inch:.0f}px/in, S≥{min_straightness_ratio:.2f})."
        analysis_message += f" (Rejected: {rejected_short_count} short, {rejected_straight_count} !straight)."

        annotated_filename = 'annotated_' + os.path.basename(mask_filepath)
        annotated_filepath = os.path.join(MASKS_FOLDER, annotated_filename)
        try:
            if cv2.imwrite(annotated_filepath, annotated_image): print(f"Saved labeled annotated image: {annotated_filepath}")
            else: print(f"Error saving annotated image: {annotated_filepath}"); annotated_filepath = None; analysis_message += " (Save err)"
        except Exception as write_e: print(f"Error saving annotated image: {write_e}"); annotated_filepath = None; analysis_message += f" (Save err: {write_e})"

        return annotated_filepath, analysis_results, analysis_message
    except Exception as e:
        print(f"Error during mask analysis: {e}\n{traceback.format_exc()}")
        return None, [], f"Analysis Error: {e}"

# --- Flask Routes ---
@app.route('/', methods=['GET'])
def index():
    """Renders the main upload page."""
    print("Route '/' accessed.")
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_and_process():
    """Handles file upload, processing, analysis and displays results."""
    print("\n--- Request received for route '/upload' (POST) ---")
    if inference_func is None: flash('Model not loaded.'); return redirect(url_for('index'))
    if 'nir_image' not in request.files: flash('No file part.'); return redirect(request.url)
    file = request.files['nir_image']
    if file.filename == '': flash('No file selected.'); return redirect(request.url)
    if not allowed_file(file.filename): flash(f'Invalid file type.'); return redirect(request.url)

    try: # Get scale factor
        scale_input = request.form.get('scale_ppi', '').strip()
        analysis_pixels_per_inch = float(scale_input) if scale_input else DEFAULT_PIXELS_PER_INCH
        if analysis_pixels_per_inch <= 0: analysis_pixels_per_inch = DEFAULT_PIXELS_PER_INCH
        print(f"Using scale factor: {analysis_pixels_per_inch} px/inch")
    except ValueError:
        analysis_pixels_per_inch = DEFAULT_PIXELS_PER_INCH
        flash(f"Using default scale: {DEFAULT_PIXELS_PER_INCH} px/inch")

    try: # Save uploaded file
        _, ext = os.path.splitext(file.filename)
        unique_filename = str(uuid.uuid4()) + ext
        original_filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(original_filepath)
        print(f"Uploaded file saved: {original_filepath}")
    except Exception as e: flash(f'Error saving file: {e}'); return redirect(url_for('index'))

    # --- Process ---
    input_tensor = preprocess_image(original_filepath, INPUT_MODEL_DIMENSION)
    if input_tensor is None: flash('Error processing image.'); return redirect(url_for('index'))

    mask_predictions = None
    try: # Inference
        results = inference_func(input_tensor)
        output_key = 'output_0'
        if output_key not in results: flash(f"Model output key '{output_key}' not found."); return redirect(url_for('index'))
        mask_predictions = results[output_key].numpy()
        print("Inference successful.")
    except Exception as e: flash(f'Inference error: {e}'); print(f"Inference error: {e}"); return redirect(url_for('index'))

    # Generate B&W Mask
    mask_filename = 'mask_' + unique_filename
    mask_filepath = os.path.join(app.config['MASKS_FOLDER'], mask_filename)
    if not generate_visual_mask(mask_predictions, mask_filepath):
        flash('Error generating visual mask.'); return redirect(url_for('index'))

    # --- Analyze Mask ---
    annotated_mask_filepath, analysis_results_list, analysis_message = analyze_mask_for_sites(mask_filepath, analysis_pixels_per_inch)

    # --- Prepare URLs ---
    annotated_mask_url = None
    if annotated_mask_filepath:
        annotated_mask_filename = os.path.basename(annotated_mask_filepath)
        try: annotated_mask_url = url_for('static', filename=f'masks/{annotated_mask_filename}')
        except Exception as e: print(f"Error generating URL for annotated mask: {e}")

    try:
        original_image_url = url_for('static', filename=f'uploads/{unique_filename}')
        mask_image_url = url_for('static', filename=f'masks/{mask_filename}')
        print(f"Prepared URLs: Original='{original_image_url}', Mask='{mask_image_url}', Annotated='{annotated_mask_url}'")
    except Exception as e: flash('Error generating URLs.'); print(f"Error creating URLs: {e}"); return redirect(url_for('index'))

    # --- Render Results ---
    print("Processing complete. Rendering results...")
    return render_template('index.html',
                           original_image=original_image_url,
                           mask_image=mask_image_url,
                           annotated_image=annotated_mask_url,
                           analysis_results=analysis_results_list,
                           analysis_message=analysis_message)

# --- Load Model & Run App ---
print("Loading model eagerly on script start...")
if not load_tf_model():
    print("FATAL: MODEL FAILED TO LOAD.")
    # exit(1) # Optional: exit if model load fails

if __name__ == '__main__':
    print("Starting Flask development server...")
    # Set debug=False for production deployment
    app.run(host='127.0.0.1', port=5000, debug=True)

