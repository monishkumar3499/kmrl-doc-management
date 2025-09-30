import os
import hashlib
import io
import logging
from concurrent.futures import ThreadPoolExecutor
from textwrap import wrap
import pickle
import asyncio
from models import ExtractedText
from sqlalchemy import select, and_

from transformers import T5ForConditionalGeneration, T5Tokenizer

import fitz
import pytesseract
from PIL import Image
import cv2
import numpy as np

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.future import select

import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

from config import config
from models import Base, Document, DepartmentContent
from preprocessor import TextPreprocessor

# ------------------ Logging ------------------ #
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("kmrl_idms")

# ------------------ Thread Pool ------------------ #
executor = ThreadPoolExecutor(max_workers=4)
text_preprocessor = TextPreprocessor(lowercase=False)

# ------------------ Database Service ------------------ #
class DatabaseService:
    def __init__(self):
        self.engine = create_async_engine(config.DATABASE_URL, echo=False, future=True)
        self.SessionLocal = sessionmaker(self.engine, expire_on_commit=False, class_=AsyncSession)

    async def create_tables(self):
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

db_service = DatabaseService()

class FileConverter:
    """Convert any uploaded file to PDF."""

    @staticmethod
    def convert_to_pdf(file_path: str) -> str:
        base, ext = os.path.splitext(file_path)
        ext = ext.lower()
        output_path = f"{base}_converted.pdf"

        if ext in [".pdf"]:
            return file_path
        elif ext in [".docx"]:
            from docx import Document as DocxDocument
            doc = DocxDocument(file_path)
            c = canvas.Canvas(output_path, pagesize=A4)
            width, height = A4
            y = height - 50
            for para in doc.paragraphs:
                if para.text.strip():
                    c.drawString(50, y, para.text)
                    y -= 15
                    if y < 50:
                        c.showPage()
                        y = height - 50
            c.save()
            return output_path
        elif ext in [".txt"]:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            c = canvas.Canvas(output_path, pagesize=A4)
            width, height = A4
            y = height - 50
            for line in lines:
                if line.strip():
                    c.drawString(50, y, line.strip())
                    y -= 15
                    if y < 50:
                        c.showPage()
                        y = height - 50
            c.save()
            return output_path
        elif ext in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
            img = Image.open(file_path)
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(output_path, "PDF")
            return output_path
        elif ext in [".xls", ".xlsx"]:
            import pandas as pd
            xls = pd.ExcelFile(file_path)
            c = canvas.Canvas(output_path, pagesize=A4)
            width, height = A4
            y = height - 50
            for sheet in xls.sheet_names:
                df = pd.read_excel(xls, sheet)
                c.setFont("Helvetica-Bold", 12)
                c.drawString(50, y, f"Sheet: {sheet}")
                y -= 20
                c.setFont("Helvetica", 10)
                for row in df.values.tolist():
                    line = " | ".join(map(str, row))
                    c.drawString(50, y, line)
                    y -= 15
                    if y < 50:
                        c.showPage()
                        y = height - 50
                c.showPage()
                y = height - 50
            c.save()
            return output_path
        else:
            raise ValueError(f"Unsupported file type: {ext}")

# ------------------ Document Processor ------------------ #
class DocumentProcessor:
    def calculate_checksum(self, file_path: str) -> str:
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def extract_metadata(self, file_path: str) -> dict:
        doc = fitz.open(file_path)
        metadata = {"pages": doc.page_count, "file_size": os.path.getsize(file_path)}
        doc.close()
        return metadata

doc_processor = DocumentProcessor()

# ------------------ OCR Service ------------------ #
import pdfplumber

class OCRService:

    def __init__(self, config):
        self.config = config
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    def preprocess_image(self, pil_image: Image.Image) -> Image.Image:
        """Convert image to grayscale, threshold, and denoise."""
        img = np.array(pil_image)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(
            gray,
            getattr(self.config, "PREPROCESS_THRESHOLD", 150),
            255,
            cv2.THRESH_BINARY
        )
        denoised = cv2.medianBlur(thresh, 3)
        return Image.fromarray(denoised)

    def extract_readable_tables(self, tess_data: dict) -> list[str]:
        """Convert pytesseract.image_to_data output into table-like lines."""
        from collections import defaultdict
        lines_dict = defaultdict(list)
        n_boxes = len(tess_data['text'])

        for i in range(n_boxes):
            word = tess_data['text'][i].strip()
            if not word:
                continue
            key = (tess_data['block_num'][i], tess_data['line_num'][i])
            lines_dict[key].append({
                "text": word,
                "left": tess_data['left'][i],
                "width": tess_data['width'][i]
            })

        table_lines = []
        for key, words in sorted(lines_dict.items()):
            if len(words) > 1 and (max(w['left'] + w['width'] for w in words) - min(w['left'] for w in words)) > 200:
                table_lines.append(" | ".join([w['text'] for w in words]))
        return table_lines

    def extract_text_and_tables(self, file_path: str) -> list[dict]:
        """
        Extract both paragraph text and tables per page.
        1. Try structured table extraction via pdfplumber
        2. Fallback to OCR for scanned PDFs
        """
        extracted_pages = []

        try:
            with pdfplumber.open(file_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    # Extract tables from structured PDF
                    tables = page.extract_tables() or []
                    text = page.extract_text() or ""
                    text = text_preprocessor.preprocess(text)

                    # Fallback to OCR if page empty
                    if not text.strip() and not tables:
                        doc = fitz.open(file_path)
                        pix = doc[i].get_pixmap(dpi=getattr(self.config, "OCR_DPI", 300))
                        img = Image.open(io.BytesIO(pix.tobytes()))
                        preprocessed_img = self.preprocess_image(img)

                        # OCR text
                        lang_config = getattr(self.config, "TESSERACT_CONFIG", "--oem 3 --psm 6 -l mal+eng")
                        ocr_text = pytesseract.image_to_string(preprocessed_img, config=lang_config)
                        text = text or text_preprocessor.preprocess(ocr_text)

                        # OCR tables
                        raw_table_data = pytesseract.image_to_data(preprocessed_img, output_type=pytesseract.Output.DICT)
                        tables = [self.extract_readable_tables(raw_table_data)]

                        doc.close()

                    extracted_pages.append({
                        "page_number": i + 1,
                        "text": text,
                        "tables": tables,
                        "confidence": 90
                    })
        except Exception as e:
            logger.warning(f"pdfplumber failed, fallback to OCR only: {e}")
            # Fallback: OCR only for all pages
            extracted_pages = self.extract_text_and_tables_ocr_only(file_path)

        return extracted_pages

    def extract_text_and_tables_ocr_only(self, file_path: str) -> list[dict]:
        """Original OCR-only method for fallback."""
        doc = fitz.open(file_path)
        extracted_pages = []

        for page_num in range(doc.page_count):
            try:
                page = doc[page_num]
                pix = page.get_pixmap(dpi=getattr(self.config, "OCR_DPI", 300))
                img_bytes = pix.tobytes()
                if not img_bytes:
                    logger.warning(f"Page {page_num+1} produced empty image. Skipping.")
                    continue

                img = Image.open(io.BytesIO(img_bytes))
                preprocessed_img = self.preprocess_image(img)

                lang_config = getattr(self.config, "TESSERACT_CONFIG", "--oem 3 --psm 6 -l mal+eng")
                text = pytesseract.image_to_string(preprocessed_img, config=lang_config)
                clean_text = text_preprocessor.preprocess(text)

                if not clean_text.strip():
                    logger.warning(f"Page {page_num+1} is empty after preprocessing. Skipping.")
                    continue

                raw_table_data = pytesseract.image_to_data(preprocessed_img, output_type=pytesseract.Output.DICT)
                table_lines = self.extract_readable_tables(raw_table_data)

                extracted_pages.append({
                    "page_number": page_num + 1,
                    "text": clean_text,
                    "tables": table_lines,
                    "confidence": 90
                })

            except Exception as e:
                logger.error(f"Error processing page {page_num+1}: {e}")
                continue

        doc.close()
        return extracted_pages

ocr_service = OCRService(config)


# ------------------ Summarizer ------------------ #
class Summarizer:
    def __init__(self, model_name="t5-small", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)

    def summarize(self, text: str, max_length=150, min_length=50) -> str:
        if not text.strip():
            return ""
        input_text = "summarize: " + text
        inputs = self.tokenizer.encode(
            input_text, return_tensors="pt", max_length=512, truncation=True
        ).to(self.device)

        summary_ids = self.model.generate(
            inputs,
            max_length=max_length,
            min_length=min_length,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

summarizer = Summarizer(model_name="t5-small")


# ------------------ BERT + Keyword Classifier ------------------ #
class DepartmentClassifier:
    def __init__(self, model_dir="bert_model/saved_model"):
        # Load tokenizer and model
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()

        # Load label encoder
        label_encoder_path = f"{model_dir}/label_encoder.pkl"
        with open(label_encoder_path, "rb") as f:
            self.label_encoder = pickle.load(f)

        # Define keywords per department
        self.DEPT_KEYWORDS = {
            "HR": [
                "leave", "recruitment", "policy", "training", "onboarding", "diversity", "employee",
                "payroll", "performance", "appraisal", "benefits", "attendance", "retention",
                "hiring", "staffing", "career", "promotion", "grievance", "resignation", "termination",
                "wellbeing", "orientation", "HRIS", "workforce", "talent"
            ],
            "Finance": [
                "budget", "expenses", "revenue", "income", "depreciation", "account", "finance", "cost",
                "profit", "loss", "balance sheet", "cash flow", "tax", "audit", "investment",
                "funding", "financial", "ledger", "capital", "billing", "invoicing", "accounting",
                "expenditure", "fiscal", "forecast", "capitalization", "ROI", "return on investment",
                "EBITDA", "gross margin", "net margin", "operating income", "EPS", "percentage", "ratio",
                "figures", "valuation", "assets", "liabilities", "equity", "interest", "dividends", "costs",
                "revenue stream", "financial statement", "numbers", "amount", "Rs.", "$", "₹"
            ],
            "Operations": [
                "maintenance", "inspection", "schedule", "operational", "assets",
                "logistics", "fleet", "transport", "supply chain", "procedure",
                "workflow", "efficiency", "monitoring", "standard operating", "downtime", "shift",
                "operations", "quality control", "process", "production", "tracking", "inventory"
            ],
            "Engineering": [
                "technical", "track", "rolling stock", "infrastructure", "engineering", "equipment",
                "design", "project", "construction", "systems", "mechanical", "electrical", "civil",
                "automation", "prototype", "installation", "testing", "commissioning", "specifications",
                "blueprint", "drawings", "repair", "upgrade", "modification", "fabrication"
            ],
            "Safety & Compliance": [
                "safety", "safety leadership", "risk assessment", "incident", "hazard",
                "emergency", "accident", "employee safety", "safety audit", "safety training",
                "compliance", "regulation", "policy compliance", "legal compliance",
                "standard", "audit report", "procedure compliance", "safety compliance",
                "environmental compliance", "regulatory", "statutory", "internal audit"
            ]
        }

    def classify_text(self, text: str) -> dict:
        # Tokenize
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        # Model prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1).squeeze()

        # Top 2 model predictions
        top2_probs, top2_idx = torch.topk(probs, k=2)
        top2_labels = [self.label_encoder.inverse_transform([i])[0] for i in top2_idx.tolist()]

        # Keyword matching
        text_lower = text.lower()
        keyword_scores = {}
        matched_keywords = {}
        for dept, keywords in self.DEPT_KEYWORDS.items():
            count = sum(1 for kw in keywords if kw.lower() in text_lower)
            keyword_scores[dept] = count
            matched_keywords[dept] = [kw for kw in keywords if kw.lower() in text_lower]

        # Decide final department
        keyword_max_dept = max(keyword_scores, key=lambda k: keyword_scores[k])
        if keyword_scores[keyword_max_dept] > 0:
            final_dept = keyword_max_dept
        else:
            final_dept = top2_labels[0]

        return {
            "top2_model_preds": list(zip(top2_labels, top2_probs.tolist())),
            "final_department": final_dept,
            "keywords_matched": matched_keywords.get(final_dept, [])
        }

classifier = DepartmentClassifier()

# ------------------ PDF Generator ------------------ #
class PDFGenerator:
    def __init__(self, config):
        self.config = config
        if self.config.MALAYALAM_FONT_PATH:
            pdfmetrics.registerFont(TTFont(self.config.MALAYALAM_FONT_NAME, self.config.MALAYALAM_FONT_PATH))

    def generate_department_pdf(self, content: str, department: str, document_id: str) -> str:
        output_path = os.path.join(self.config.OUTPUT_DIR, f"{document_id}_{department}.pdf")
        c = canvas.Canvas(output_path, pagesize=A4)
        width, height = A4
        c.setFont(self.config.MALAYALAM_FONT_NAME or "Helvetica", 12)
        textobject = c.beginText(50, height - 50)
        max_chars = 80
        for line in content.splitlines():
            for wrapped_line in wrap(line, max_chars):
                textobject.textLine(wrapped_line)
        c.drawText(textobject)
        c.showPage()
        c.save()
        return output_path

pdf_generator = PDFGenerator(config)


# ------------------ process_document (Updated with Summarization) ------------------ #
async def process_document(file_path: str, document_id: str):
    async with db_service.SessionLocal() as session:
        doc = None
        try:
            logger.info(f"Processing document: {document_id}")

            # Fetch document
            result = await session.execute(select(Document).where(Document.id == document_id))
            doc = result.scalar_one_or_none()
            if not doc:
                logger.error(f"Document {document_id} not found!")
                return

            # Metadata & checksum
            metadata = await asyncio.get_event_loop().run_in_executor(
                executor, doc_processor.extract_metadata, file_path
            )
            checksum = await asyncio.get_event_loop().run_in_executor(
                executor, doc_processor.calculate_checksum, file_path
            )

            doc.pages = metadata.get("pages", None)
            doc.file_size = metadata.get("file_size", None)
            doc.checksum = checksum
            doc.status = "processing"
            doc.original_path = file_path
            await session.commit()

            # OCR / PDF extraction
            ocr_results = await asyncio.get_event_loop().run_in_executor(
                executor, ocr_service.extract_text_and_tables, file_path
            )
            if not ocr_results:
                logger.warning(f"No text extracted from document {document_id}")
                doc.status = "failed"
                await session.commit()
                return

            all_dept_contents = {}  # {department: [lines]}
            extracted_entries = []  # batch DB inserts

            for page in ocr_results:
                page_num = page["page_number"]
                text = page.get("text", "")
                tables = page.get("tables", [])

                # Save full page text + tables
                extracted_entries.append(
                    ExtractedText(
                        document_id=document_id,
                        page_number=page_num,
                        text_content=text,
                        confidence=1.0,
                        ocr_engine="your_ocr_engine_name",
                        bbox=page.get("bbox", {}),
                        tables=tables
                    )
                )

                # Split text into paragraphs
                paragraphs = text.split("\n\n")
                for para in paragraphs:
                    if not para.strip():
                        continue

                    # Summarize paragraph
                    summary_text = await asyncio.get_event_loop().run_in_executor(
                        executor, summarizer.summarize, para
                    )
                    if not summary_text:
                        summary_text = para

                    # Classify paragraph
                    predictions = await asyncio.get_event_loop().run_in_executor(
                        executor, classifier.classify_text, para
                    )
                    final_dept = predictions["final_department"]
                    keywords_matched = predictions["keywords_matched"]

                    # Log paragraph assignment
                    logger.info(
                        f"[Doc {document_id} | Page {page_num}] Paragraph assigned to {final_dept} "
                        f"(keywords matched: {keywords_matched})"
                    )

                    # Append summarized lines to department content
                    if final_dept not in all_dept_contents:
                        all_dept_contents[final_dept] = []

                    for line in summary_text.splitlines():
                        all_dept_contents[final_dept].append(f"{line} [Page {page_num}]")

                    # Append table lines
                    for table in tables:
                        if isinstance(table, list):
                            if all(isinstance(row, list) for row in table):
                                table_lines = [" | ".join(map(str, row)) for row in table]
                                table_text = "\n".join(table_lines)
                            else:
                                table_text = "\n".join(map(str, table))
                            all_dept_contents[final_dept].append(f"{table_text} [Page {page_num}]")

                    # Store DepartmentContent with approved=False and priority from document
                    extracted_entries.append(
                        DepartmentContent(
                            document_id=document_id,
                            department=final_dept,
                            content=summary_text,
                            page_start=page_num,
                            page_end=page_num,
                            confidence=1.0,
                            keywords_matched=keywords_matched,
                            pdf_path=None,
                            doc_priority=doc.priority,
                            approved=False
                        )
                    )

            # Bulk insert all ExtractedText + DepartmentContent
            session.add_all(extracted_entries)
            await session.commit()

            # Generate PDFs per department asynchronously
            for dept, lines in all_dept_contents.items():
                full_text = "\n".join(lines)
                pdf_path = await asyncio.get_event_loop().run_in_executor(
                    executor, pdf_generator.generate_department_pdf, full_text, dept, document_id
                )

                # Update DepartmentContent PDF paths
                result = await session.execute(
                    select(DepartmentContent).where(
                        and_(
                            DepartmentContent.document_id == document_id,
                            DepartmentContent.department == dept
                        )
                    )
                )
                dept_entries = result.scalars().all()
                for entry in dept_entries:
                    entry.pdf_path = pdf_path
                    session.add(entry)  # important for async update
                await session.commit()

            # Update overall document approval based on department approvals
            result_all_depts = await session.execute(
                select(DepartmentContent).where(DepartmentContent.document_id == document_id)
            )
            all_depts = result_all_depts.scalars().all()
            doc.approved = all([d.approved for d in all_depts]) if all_depts else False
            doc.status = "completed"
            session.add(doc)
            await session.commit()
            logger.info(f"Document processing completed: {document_id}")

        except Exception as e:
            logger.error(f"Error processing document {document_id}: {e}")
            if doc:
                doc.status = "failed"
                session.add(doc)
                await session.commit()
