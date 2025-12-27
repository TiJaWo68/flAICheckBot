"""
Environment check script to verify Python dependencies and CUDA availability for the AI engine.

@author TiJaWo68 in cooperation with Gemini 3 Flash using Antigravity
"""
import importlib
import sys

required_packages = [
    "fastapi",
    "uvicorn",
    "PIL",
    "torch",
    "transformers",
    "peft",
    "accelerate"
]

def check_env():
    print("--- flAICheckBot Environment Check ---\n")
    print(f"Python Version: {sys.version}")
    
    missing = []
    for pkg in required_packages:
        try:
            mod = importlib.import_module(pkg.lower() if pkg != "PIL" else "PIL")
            version = getattr(mod, "__version__", "unknown")
            print(f"[OK] {pkg} (version: {version})")
        except ImportError:
            print(f"[ERROR] {pkg} is NOT installed.")
            missing.append(pkg)

    print("\n--- Hardware Check ---")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"[OK] CUDA is available (GPU: {torch.cuda.get_device_name(0)})")
            print(f"CUDA Version: {torch.version.cuda}")
        else:
            print("[INFO] CUDA is NOT available. AI tasks will run on CPU (Slow!).")
    except ImportError:
        print("[ERROR] Hardware check skipped: torch is not installed.")

    if missing:
        print(f"\nBitte installiere die fehlenden Pakete: pip install {' '.join(missing)}")
        return False
    
    print("\n[SUCCESS] Deine Umgebung ist bereit für die Entwicklung!")
    return True

if __name__ == "__main__":
    check_env()
