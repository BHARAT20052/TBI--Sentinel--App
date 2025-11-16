import cv2
import numpy as np

def segment_image(image_path: str) -> dict:
    """
    Performs real image segmentation using OpenCV to detect and quantify anomalies
    based on pixel intensity thresholding.
    """
    print(f"Analyzing MRI scan: {image_path} with OpenCV...")
    
    # 1. Load the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Could not load image.")
        return {"volume_percent": 0.0, "anomalies_detected": False}

    # 2. Define a Region of Interest (ROI) to approximate the brain area
    # Simple cropping example for a square MRI image (adjust as needed)
    rows, cols = img.shape
    crop_size = min(rows, cols)
    img_cropped = img[0:crop_size, 0:crop_size]
    
    # 3. Anomaly Detection using Thresholding
    # Set a high threshold (e.g., 200) to target bright areas (like edema/hemorrhage)
    _, thresh = cv2.threshold(img_cropped, 200, 255, cv2.THRESH_BINARY)
    
    # 4. Quantification
    # Count the number of white pixels (anomaly)
    anomaly_pixels = np.sum(thresh == 255)
    
    # Total pixels in the analyzed area
    total_pixels = img_cropped.size
    
    # Calculate the volume percentage
    if total_pixels > 0:
        volume_percent = round((anomaly_pixels / total_pixels) * 100, 2)
    else:
        volume_percent = 0.0

    anomalies_detected = volume_percent > 0.1 # Threshold set at 0.1%
    
    return {
        "volume_percent": volume_percent,
        "anomalies_detected": anomalies_detected
    }