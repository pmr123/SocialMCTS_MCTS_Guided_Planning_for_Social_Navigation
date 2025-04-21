import cv2
import numpy as np

def detect_red_blue_boxes(image):
    """
    Detect red cubes and blue cuboids in a given RGB image using classical computer vision.

    Args:
        image (np.ndarray): RGB image of shape (60, 80, 3)

    Returns:
        dict: {
            'red_cubes': int,
            'blue_cuboids': int,
            'objects': list of dicts with keys: color, shape, bbox (x, y, w, h)
        }
    """
    # Convert RGB to BGR for OpenCV compatibility
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # Convert to HSV color space for easier color filtering
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # Define color masks
    # Red (2 ranges due to HSV wrap-around)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)

    # Blue
    lower_blue = np.array([100, 100, 50])
    upper_blue = np.array([130, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    results = {
        'red_cubes': 0,
        'blue_cuboids': 0,
        'objects': []
    }

    def analyze_contours(mask, color):
        nonlocal results
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 50:  # Filter small noise
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h
            shape = "cube" if 0.8 <= aspect_ratio <= 1.2 else "cuboid"

            if color == "red" and shape == "cube":
                results['red_cubes'] += 1
            elif color == "blue" and shape == "cuboid":
                results['blue_cuboids'] += 1

            results['objects'].append({
                'color': color,
                'shape': shape,
                'bbox': (x, y, w, h)
            })

    analyze_contours(mask_red, "red")
    analyze_contours(mask_blue, "blue")

    return results


# Example output
# Make sure that the fields are always present. They may have 0 values.

#{
#    'red_cubes': 1,
#    'blue_cuboids': 2,
#    'objects': [
#        {'color': 'red', 'shape': 'cube', 'bbox': (x1, y1, w1, h1)},
#        {'color': 'blue', 'shape': 'cuboid', 'bbox': (x2, y2, w2, h2)},
#        {'color': 'blue', 'shape': 'cuboid', 'bbox': (x3, y3, w3, h3)}
#    ]
#}
