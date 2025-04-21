import cv2
import numpy as np

def detect_humans_and_goal(image):
    """
    Detect humans (blue cuboids) and goal (red cube) in a given RGB image using 
    classical computer vision.

    Args:
        image (np.ndarray): RGB image 

    Returns:
        dict: {
            'red_cubes': int,
            'blue_cuboids': int,
            'humans_detected': int,
            'objects': list of dicts with keys: color, shape, bbox (x, y, w, h)
        }
    """
    # Convert RGB to BGR for OpenCV compatibility
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # Convert to HSV color space for easier color filtering
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # Define color masks
    # Red (2 ranges due to HSV wrap-around)
    lower_red1 = np.array([0, 100, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 100, 50])
    upper_red2 = np.array([180, 255, 255])
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)

    # Blue
    lower_blue = np.array([100, 100, 50])
    upper_blue = np.array([130, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    results = {
        'red_cubes': 0,
        'blue_cuboids': 0,
        'humans_detected': 0,
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
                results['humans_detected'] += 1

            results['objects'].append({
                'color': color,
                'shape': shape,
                'bbox': (x, y, w, h)
            })

    analyze_contours(mask_red, "red")
    analyze_contours(mask_blue, "blue")

    return results

def create_tagged_image(image, detection_results):
    """
    Create a tagged version of the image with bounding boxes and labels.
    
    Args:
        image (np.ndarray): Original RGB image
        detection_results (dict): Detection results from detect_humans_and_goal
        
    Returns:
        np.ndarray: Tagged RGB image
    """
    # Create a copy of the image to draw on
    tagged_image = image.copy()
    
    # Convert to BGR for OpenCV drawing
    tagged_image_bgr = cv2.cvtColor(tagged_image, cv2.COLOR_RGB2BGR)
    
    # Draw bounding boxes and labels for each detected object
    for obj in detection_results['objects']:
        x, y, w, h = obj['bbox']
        color = (255, 0, 0) if obj['color'] == 'blue' else (0, 0, 255)  # BGR format
        label = f"{obj['color']} {obj['shape']}"
        
        # Draw rectangle
        cv2.rectangle(tagged_image_bgr, (x, y), (x + w, y + h), color, 2)
        
        # Draw label with a filled background
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(tagged_image_bgr, (x, y - text_size[1] - 5), (x + text_size[0], y), color, -1)
        cv2.putText(tagged_image_bgr, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Convert back to RGB
    tagged_image = cv2.cvtColor(tagged_image_bgr, cv2.COLOR_BGR2RGB)
    
    return tagged_image 