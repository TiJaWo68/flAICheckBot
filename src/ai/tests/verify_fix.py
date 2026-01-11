
import sys
import os
import cv2
import numpy as np

# Adjust path so we can import app
current_dir = os.path.dirname(os.path.abspath(__file__))
src_ai_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(src_ai_dir)

try:
    from app.preprocessing import segment_lines
    print("Import successful.")
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

def test_segment_lines():
    # Create a dummy image (white background with some black lines)
    img = np.ones((500, 500, 3), dtype=np.uint8) * 255
    
    # Draw some "text" lines
    for i in range(50, 450, 50):
        cv2.line(img, (50, i), (450, i), (0, 0, 0), 2)
        # Add some random noise/text-like pixels
        for x in range(50, 450, 5):
             if x % 10 == 0:
                 cv2.circle(img, (x, i), 2, (0,0,0), -1)

    print("Running segment_lines on dummy image...")
    try:
        lines = segment_lines(img)
        print(f"Success! Found {len(lines)} lines.")
        for i, (l_img, bbox, meta) in enumerate(lines):
            print(f"Line {i}: bbox={bbox}, density={meta.get('density')}, rejected={meta.get('rejected')}")
    except NameError as e:
        print(f"Caught NameError: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Caught unexpected exception: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_segment_lines()
