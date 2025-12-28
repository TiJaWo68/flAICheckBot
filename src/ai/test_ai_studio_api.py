import requests
import sys
import os

"""
Automated test for AI Studio API Key authentication.
Usage: python test_ai_studio_api.py <YOUR_API_KEY>
"""

def test_grading_with_api_key(api_key):
    url = "http://127.0.0.1:8000/grade_vertex"
    
    payload = {
        'task': 'Aufgabe 1: Was ist 2+2? (Max: 5 Punkte)',
        'expected': '4',
        'actual': 'Das Ergebnis ist 4',
        'apiKey': api_key
    }
    
    print(f"Sending test request to {url} using API Key...")
    try:
        response = requests.post(url, data=payload)
        print(f"Status Code: {response.status_code}")
        
        result = response.json()
        print("Response JSON:")
        import json
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        if result.get("status") == "success":
            print("\n✅ AI Studio Path (API Key) is working correctly!")
        else:
            print(f"\n❌ AI Studio Path failed: {result.get('message')}")
            
    except Exception as e:
        print(f"\n❌ Request failed: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: Please provide your Gemini API Key as an argument.")
        print("Usage: python test_ai_studio_api.py <API_KEY>")
        sys.exit(1)
        
    test_grading_with_api_key(sys.argv[1])
