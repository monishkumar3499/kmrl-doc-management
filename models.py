import uuid
from datetime import datetime
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy import Column, String, Integer, DateTime, Text, Float, JSON, ForeignKey, Boolean
from sqlalchemy.dialects.postgresql import UUID

Base = declarative_base()

# ------------------ Document Table ------------------ #
class Document(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    original_path = Column(String, nullable=True)
    file_size = Column(Integer)
    checksum = Column(String)
    pages = Column(Integer)
    status = Column(String, default="uploaded")  # Pending / Completed / Failed
    priority = Column(String, default="Low")
    approved = Column(Boolean, default=False)   # overall approval status
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    doc_metadata = Column(JSON, default=dict)

    extracted_texts = relationship(
        "ExtractedText", back_populates="document", cascade="all, delete-orphan"
    )
    department_contents = relationship(
        "DepartmentContent", back_populates="document", cascade="all, delete-orphan"
    )

# ------------------ Department Content Table ------------------ #
class DepartmentContent(Base):
    __tablename__ = "department_contents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    department = Column(String, nullable=False)
    content = Column(Text)
    page_start = Column(Integer)
    page_end = Column(Integer)
    confidence = Column(Float, default=1.0)
    keywords_matched = Column(JSON, default=dict)
    pdf_path = Column(String)
    doc_priority = Column(String, nullable=False, default="Low")
    approved = Column(Boolean, default=False)   # department approval
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    document = relationship("Document", back_populates="department_contents")

# ------------------ Extracted Text Table ------------------ #
class ExtractedText(Base):
    __tablename__ = "extracted_texts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    page_number = Column(Integer, nullable=False)
    text_content = Column(Text)
    language = Column(String, default="unknown")
    confidence = Column(Float, default=0.0)
    ocr_engine = Column(String, default="tesseract")
    bbox = Column(JSON, default=dict)
    tables = Column(JSON, default=dict)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    document = relationship("Document", back_populates="extracted_texts")


