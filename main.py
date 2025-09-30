import logging
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes import router          # absolute import
from services import db_service    # absolute import

# ------------------ Logging ------------------ #
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("kmrl_idms")

# ------------------ FastAPI App ------------------ #
app = FastAPI(title="KMRL IDMS Phase 1", version="1.0.1")

# ------------------ CORS Middleware ------------------ #
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:3000"],  # frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ Include Routes ------------------ #
app.include_router(router)

# ------------------ Startup Event ------------------ #
@app.on_event("startup")
async def startup():
    await db_service.create_tables()
    logger.info("KMRL IDMS Phase 1 started")

# ------------------ Run ------------------ #
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
