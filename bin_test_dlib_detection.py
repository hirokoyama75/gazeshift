#!/usr/bin/env python3
"""
Dlib Face Detection Test

Test application to visualize dlib 68-point face landmarks detection.

Usage:
    python bin_test_dlib_detection.py             # Use default camera
    python bin_test_dlib_detection.py --camera 1  # Use camera device 1

Controls:
    - 'b': Toggle background on/off
    - 'q': Quit
"""

import cv2
import numpy as np
import dlib


def detect_camera_resolution(camera_id: int) -> tuple[int, int]:
    """
    Detect the actual resolution of the specified camera.
    
    Args:
        camera_id: Camera device ID
        
    Returns:
        Tuple of (width, height) in pixels
    """
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Warning: Could not open camera {camera_id}, using default resolution")
        return (640, 480)
    
    # Get the actual resolution
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    print(f"Detected camera resolution: {width}x{height}")
    return (width, height)


def draw_landmarks(image: np.ndarray, shape, color=(0, 255, 0), radius=2):
    """
    Draw all 68 facial landmarks on the image.
    
    Args:
        image: Image to draw on
        shape: dlib shape object with 68 landmarks
        color: Color for landmark points (BGR)
        radius: Radius of landmark circles
    """
    for i in range(68):
        x = shape.part(i).x
        y = shape.part(i).y
        cv2.circle(image, (x, y), radius, color, -1)


def draw_face_contours(image: np.ndarray, shape, color=(0, 255, 0), thickness=1):
    """
    Draw face contours connecting landmarks.
    
    Args:
        image: Image to draw on
        shape: dlib shape object with 68 landmarks
        color: Color for contour lines (BGR)
        thickness: Line thickness
    """
    # Jaw line (0-16)
    for i in range(16):
        pt1 = (shape.part(i).x, shape.part(i).y)
        pt2 = (shape.part(i + 1).x, shape.part(i + 1).y)
        cv2.line(image, pt1, pt2, color, thickness)
    
    # Right eyebrow (17-21)
    for i in range(17, 21):
        pt1 = (shape.part(i).x, shape.part(i).y)
        pt2 = (shape.part(i + 1).x, shape.part(i + 1).y)
        cv2.line(image, pt1, pt2, color, thickness)
    
    # Left eyebrow (22-26)
    for i in range(22, 26):
        pt1 = (shape.part(i).x, shape.part(i).y)
        pt2 = (shape.part(i + 1).x, shape.part(i + 1).y)
        cv2.line(image, pt1, pt2, color, thickness)
    
    # Nose bridge (27-30)
    for i in range(27, 30):
        pt1 = (shape.part(i).x, shape.part(i).y)
        pt2 = (shape.part(i + 1).x, shape.part(i + 1).y)
        cv2.line(image, pt1, pt2, color, thickness)
    
    # Nose bottom (31-35)
    for i in range(31, 35):
        pt1 = (shape.part(i).x, shape.part(i).y)
        pt2 = (shape.part(i + 1).x, shape.part(i + 1).y)
        cv2.line(image, pt1, pt2, color, thickness)
    
    # Right eye (36-41)
    for i in range(36, 41):
        pt1 = (shape.part(i).x, shape.part(i).y)
        pt2 = (shape.part(i + 1).x, shape.part(i + 1).y)
        cv2.line(image, pt1, pt2, color, thickness)
    # Close right eye
    pt1 = (shape.part(41).x, shape.part(41).y)
    pt2 = (shape.part(36).x, shape.part(36).y)
    cv2.line(image, pt1, pt2, color, thickness)
    
    # Left eye (42-47)
    for i in range(42, 47):
        pt1 = (shape.part(i).x, shape.part(i).y)
        pt2 = (shape.part(i + 1).x, shape.part(i + 1).y)
        cv2.line(image, pt1, pt2, color, thickness)
    # Close left eye
    pt1 = (shape.part(47).x, shape.part(47).y)
    pt2 = (shape.part(42).x, shape.part(42).y)
    cv2.line(image, pt1, pt2, color, thickness)
    
    # Outer mouth (48-59)
    for i in range(48, 59):
        pt1 = (shape.part(i).x, shape.part(i).y)
        pt2 = (shape.part(i + 1).x, shape.part(i + 1).y)
        cv2.line(image, pt1, pt2, color, thickness)
    # Close outer mouth
    pt1 = (shape.part(59).x, shape.part(59).y)
    pt2 = (shape.part(48).x, shape.part(48).y)
    cv2.line(image, pt1, pt2, color, thickness)
    
    # Inner mouth (60-67)
    for i in range(60, 67):
        pt1 = (shape.part(i).x, shape.part(i).y)
        pt2 = (shape.part(i + 1).x, shape.part(i + 1).y)
        cv2.line(image, pt1, pt2, color, thickness)
    # Close inner mouth
    pt1 = (shape.part(67).x, shape.part(67).y)
    pt2 = (shape.part(60).x, shape.part(60).y)
    cv2.line(image, pt1, pt2, color, thickness)


def run_face_detection(camera_id: int, model_path: str = './lm_feat/shape_predictor_68_face_landmarks.dat'):
    """
    Run dlib face detection and visualization.
    
    Args:
        camera_id: Camera device ID
        model_path: Path to dlib shape predictor model
    """
    # Detect camera resolution
    video_size = detect_camera_resolution(camera_id)
    
    # Setup dlib
    print(f"Loading dlib model from: {model_path}")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(model_path)

    print(f"Starting camera {camera_id}...")
    print("Press 'b' to toggle background, 'q' to quit")
    
    # State
    show_background = True

    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, video_size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, video_size[1])

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Cannot receive frame")
            break

        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = detector(gray, 0)

        # Use original frame or black background based on toggle
        if show_background:
            annotated_image = np.copy(frame)
        else:
            annotated_image = np.zeros_like(frame)

        # Loop through the detected faces to visualize
        for face in faces:
            # Predict landmarks
            shape = predictor(gray, face)
            
            # Draw face bounding box
            x1, y1 = face.left(), face.top()
            x2, y2 = face.right(), face.bottom()
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # Draw face contours
            draw_face_contours(annotated_image, shape, color=(0, 255, 0), thickness=1)
            
            # Draw landmarks
            draw_landmarks(annotated_image, shape, color=(0, 255, 255), radius=2)

        cv2.imshow('Dlib Face Landmarker', annotated_image)
        
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('b'):
            show_background = not show_background
            status = "ON" if show_background else "OFF"
            print(f"Background: {status}")

    cap.release()
    cv2.destroyAllWindows()
    print("Shutdown complete")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Dlib Face Detection Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Controls:
  'b' - Toggle background on/off
  'q' - Quit the application

Examples:
  %(prog)s                  # Use default camera
  %(prog)s --camera 1       # Use camera device 1
        """,
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device ID (default: 0)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="./lm_feat/shape_predictor_68_face_landmarks.dat",
        help="Path to dlib shape predictor model (default: ./lm_feat/shape_predictor_68_face_landmarks.dat)",
    )
    args = parser.parse_args()

    run_face_detection(args.camera, args.model)


if __name__ == "__main__":
    main()
