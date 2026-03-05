from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from app.routers import predicting
from app.config import settings

app = FastAPI(title=settings.PROJECT_NAME, version=settings.VERSION)

# Enable CORS for local testing (adjust origins for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# include routers
app.include_router(predicting.router, prefix="/api")


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"detail": str(exc)})


@app.get("/")
def root():
    return {"message": "Satellite image prediction service is up."}
