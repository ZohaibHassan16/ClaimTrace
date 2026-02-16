from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from app.engine import NewsVerifier
import uvicorn

app = FastAPI(title="ClaimTrace API", version="1.0.0")
templates = Jinja2Templates(directory="templates")

verifier = None

@app.on_event("startup")
async def load_models():
    global verifier
    verifier = NewsVerifier()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze")
async def analyze(file: UploadFile = File(...), caption: str = Form(...)):
    contents = await file.read()
    results = verifier.verify(contents, caption)
    return results

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)