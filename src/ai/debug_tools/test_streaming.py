import requests
import json
import sys

def test_streaming(image_path, language="de"):
    url = "http://127.0.0.1:8000/recognize"
    files = {"file": open(image_path, "rb")}
    data = {"language": language, "preprocess": "true"}
    
    print(f"Testing streaming OCR for {image_path}...")
    try:
        response = requests.post(url, files=files, data=data, stream=True)
    except Exception as e:
        print(f"Connection failed: {e}")
        return
    
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.text)
        return

    for line in response.iter_lines():
        if line:
            line_str = line.decode('utf-8')
            try:
                obj = json.loads(line_str)
                if 'type' in obj:
                    print(f"Received: {obj['type']}")
                    if obj['type'] == 'line':
                        print(f"  Line {obj['index']}/{obj['total']}: {obj['text']} (BBox: {obj['bbox']})")
                    elif obj['type'] == 'final':
                        print(f"  Final Text: {obj['text'][:50]}...")
                else:
                    print(f"Unexpected JSON (no type): {obj}")
            except Exception as e:
                print(f"Raw line (not JSON?): {line_str}")
                print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_streaming.py <image_path> [language]")
    else:
        lang = sys.argv[2] if len(sys.argv) > 2 else "de"
        test_streaming(sys.argv[1], lang)
