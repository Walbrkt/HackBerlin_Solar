from __future__ import annotations

import importlib
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates


processor = importlib.import_module("3d_processor")

BASE_DIR = Path(__file__).resolve().parent
MAX_UPLOAD_BYTES = 200 * 1024 * 1024

app = FastAPI(
    title="AI Solar Roof Designer",
    description="Detect flat roof surfaces from GLB building models.",
    version="1.0.0",
)
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload-model/")
async def upload_model(file: UploadFile = File(...)) -> JSONResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file was uploaded.")

    suffix = Path(file.filename).suffix.lower()
    if suffix != ".glb":
        raise HTTPException(status_code=400, detail="Only binary .glb model files are supported.")

    temp_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".glb") as temp_file:
            temp_path = Path(temp_file.name)
            uploaded_size = 0

            while chunk := await file.read(1024 * 1024):
                uploaded_size += len(chunk)
                if uploaded_size > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="The uploaded model is larger than 200 MB.")
                temp_file.write(chunk)

        result: dict[str, Any] = processor.process_glb_model(temp_path)
        return JSONResponse(content=result)

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model processing failed: {exc}") from exc
    finally:
        await file.close()
        if temp_path and temp_path.exists():
            temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
