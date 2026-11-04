"""API skeleton: health works; recognizer backend has not been integrated."""

from fastapi import FastAPI, HTTPException, UploadFile

app = FastAPI(title="Vi-DFD", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "scaffold", "model_loaded": False}


@app.post("/recognize")
async def recognize(image: UploadFile):
    raise HTTPException(status_code=501, detail="ONNX recognizer is not implemented yet")
