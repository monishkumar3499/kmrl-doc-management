import os
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

@dataclass
class Config:
    # ---------------- Database ----------------
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql+asyncpg://postgres:password@localhost:5432/kmrl_idms"
    )

    # ---------------- Directories ----------------
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "./uploads")
    OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", "./output")

    # ---------------- OCR ----------------
    TESSERACT_CONFIG: str = os.getenv("TESSERACT_CONFIG", "--oem 3 --psm 6 -l eng+mal")
    OCR_DPI: int = int(os.getenv("OCR_DPI", "300"))
    PREPROCESS_THRESHOLD: int = int(os.getenv("PREPROCESS_THRESHOLD", "150"))

    # ---------------- Malayalam Font ----------------
    MALAYALAM_FONT_PATH: str = os.getenv(
        "MALAYALAM_FONT_PATH",
        "./fonts/NotoSansMalayalam-Regular.ttf"
    )
    MALAYALAM_FONT_NAME: str = os.getenv("MALAYALAM_FONT_NAME", "NotoMalayalam")

    # ---------------- DistilBERT Model ----------------
    DEPT_MODEL_DIR: str = os.getenv("DEPT_MODEL_DIR", "bert_model/saved_model")  # DistilBERT path
    DEPT_THRESHOLD: float = float(os.getenv("DEPT_THRESHOLD", 0.2))              # optional
    USE_KEYWORDS: bool = os.getenv("USE_KEYWORDS", "True").lower() == "true"     # Enable keyword mapping
    KEYWORD_WEIGHT: float = float(os.getenv("KEYWORD_WEIGHT", 1.0))              # not used currently

# ---------------- Ensure directories exist ----------------
config = Config()
for dir_path in [config.UPLOAD_DIR, config.OUTPUT_DIR]:
    Path(dir_path).mkdir(parents=True, exist_ok=True)
