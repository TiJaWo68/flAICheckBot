import uvicorn
from contextlib import asynccontextmanager

if __name__ == "__main__":
    # If run directly
    uvicorn.run("app.api:app", host="0.0.0.0", port=8000, reload=False)
