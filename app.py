import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

# Load the trained YOLOv8 model
model = YOLO('yolov8_parking_spot_model2original.pt')

def process_image_with_black_background(image, boxes):
    """
    Processes an image and overlays bounding boxes on a black background.

    Args:
        image: Input image (numpy array).
        boxes: Detected bounding boxes and class labels.

    Returns:
        black_background_image: Image with black background and bounding boxes.
    """
    height, width, _ = image.shape
    black_background_image = np.zeros((height, width, 3), dtype=np.uint8)

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
        confidence = box.conf.item()  # Confidence score
        class_id = int(box.cls.item())  # Class ID: 0 for empty, 1 for occupied

        color = (0, 255, 0) if class_id == 0 else (255, 0, 0)
        # label = f"{'Empty' if class_id == 0 else 'Occupied'} {confidence:.2f}"

        cv2.rectangle(black_background_image, (x1, y1), (x2, y2), color, 2)
        # cv2.putText(black_background_image, label, (x1, y1 - 10),
                    # cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return black_background_image

def process_image(image, boxes):
    """
    Enhances the input image and overlays bounding boxes.

    Args:
        image: Input image (numpy array).
        boxes: Detected bounding boxes and class labels.

    Returns:
        final_image: Enhanced image with bounding boxes.
    """
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened_image = cv2.filter2D(image, -1, kernel)

    def adjust_brightness_contrast(image, brightness=50, contrast=1.5):
        return cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)

    enhanced_image = adjust_brightness_contrast(sharpened_image)

    hsv_image = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2HSV)
    hsv_image[..., 1] = np.clip(hsv_image[..., 1] * 1.5, 0, 255)
    final_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
        confidence = box.conf.item()
        class_id = int(box.cls.item())
        color = (0, 255, 0) if class_id == 0 else (255, 0, 0)
        label = f"{'Empty' if class_id == 0 else 'Occupied'} {confidence:.2f}"

        cv2.rectangle(final_image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(final_image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return final_image

# Streamlit UI
st.title("Parking Spot Detection")

uploaded_image = st.file_uploader("Upload an image for detection", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Display Original Image
    st.subheader("Original Image")
    st.image(image, caption="Original Image", use_column_width=True)

    results = model.predict(source=image)
    result = results[0]
    boxes = result.boxes

    # Process the enhanced image
    processed_enhanced_image = process_image(image, boxes)

    # Process the image with a black background
    processed_black_background = process_image_with_black_background(image, boxes)

    # Display Enhanced Image
    st.subheader("Enhanced Image with Bounding Boxes")
    st.image(processed_enhanced_image, caption="Enhanced Image Output", use_column_width=True)

    # Display Black Background Output
    st.subheader("Black Background with Bounding Boxes")
    st.image(processed_black_background, caption="Black Background Output", use_column_width=True)

    # Count and display parking spots
    classes = boxes.cls
    num_empty_spots = (classes == 0).sum().item()
    num_occupied_spots = (classes == 1).sum().item()
    st.write(f"Number of empty parking spots: {num_empty_spots}")
    st.write(f"Number of occupied parking spots: {num_occupied_spots}")
