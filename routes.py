import os
import uuid
from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy import select, and_
from services import db_service, process_document
from models import Document, DepartmentContent, ExtractedText
from services import ocr_service

router = APIRouter()

# ------------------ Upload Document ------------------ #
@router.post("/api/v1/upload")
async def upload_document(
    file: UploadFile = File(...),
    priority: str = Form("Low"),
    background_tasks: BackgroundTasks = None
):
    if priority not in ["Low", "High"]:
        priority = "Low"

    async with db_service.SessionLocal() as session:
        existing_doc = await session.execute(
            select(Document).where(Document.filename == file.filename)
        )
        existing_doc = existing_doc.scalar_one_or_none()
        if existing_doc:
            raise HTTPException(
                status_code=400,
                detail=f"A document with the name '{file.filename}' already exists."
            )

        os.makedirs("uploads", exist_ok=True)
        file_path = os.path.join("uploads", file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())

        new_doc = Document(
            filename=file.filename,
            file_path=file_path,
            status="uploaded",
            priority=priority
        )
        session.add(new_doc)
        await session.commit()

        if background_tasks:
            background_tasks.add_task(process_document, file_path, new_doc.id)

    return {
        "id": str(new_doc.id),
        "filename": new_doc.filename,
        "status": "processing",
        "priority": new_doc.priority
    }

# ------------------ Get Page-wise OCR Tables ------------------ #
@router.get("/api/v1/documents/{doc_id}/ocr-tables")
async def get_pagewise_ocr_tables(doc_id: str):
    try:
        doc_uuid = uuid.UUID(doc_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document_id format")

    async with db_service.SessionLocal() as session:
        result_doc = await session.execute(select(Document).where(Document.id == doc_uuid))
        doc = result_doc.scalar_one_or_none()
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")

        result_text = await session.execute(
            select(ExtractedText)
            .where(ExtractedText.document_id == doc_uuid)
            .order_by(ExtractedText.page_number.asc())
        )
        ocr_entries = result_text.scalars().all()
        if not ocr_entries:
            raise HTTPException(status_code=404, detail="OCR text not found")

        def normalize_table(table):
            normalized = []
            for row in table:
                if isinstance(row, list):
                    normalized.append([str(cell) for cell in row])
                else:
                    normalized.append([str(row)])
            return normalized

        pages = []
        for entry in ocr_entries:
            structured_tables = []
            if entry.tables:
                for table in entry.tables:
                    structured_tables.append({"rows": normalize_table(table)})

            pages.append({
                "page_number": entry.page_number,
                "tables": structured_tables
            })

        return {
            "document_id": str(doc.id),
            "filename": doc.filename,
            "pages": pages
        }

# ------------------ List All Documents with Approval Status ------------------ #
@router.get("/api/v1/documents-list")
async def list_all_documents():
    async with db_service.SessionLocal() as session:
        result = await session.execute(
            select(Document).order_by(Document.created_at.desc())
        )
        docs = result.scalars().all()

        doc_list = []
        for doc in docs:
            dept_result = await session.execute(
                select(DepartmentContent).where(DepartmentContent.document_id == doc.id)
            )
            dept_contents = dept_result.scalars().all()

            dept_approvals = {d.department: getattr(d, "approved", False) for d in dept_contents}
            assigned_approvals = [approved for approved in dept_approvals.values() if approved is not None]
            overall_approved = all(assigned_approvals) if assigned_approvals else False

            if doc.approved != overall_approved:
                doc.approved = overall_approved
                session.add(doc)
                await session.commit()

            doc_list.append({
                "id": str(doc.id),
                "filename": doc.filename,
                "status": doc.status,
                "priority": doc.priority,
                "approved": overall_approved,
                "departments_approval": dept_approvals,
                "created_at": doc.created_at.strftime("%Y-%m-%d")
            })

        return doc_list

# ------------------ Toggle Department Approval ------------------ #
@router.post("/api/v1/departments/{doc_id}/{department}/approve")
async def toggle_department_approval(doc_id: str, department: str):
    try:
        doc_uuid = uuid.UUID(doc_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document_id")

    async with db_service.SessionLocal() as session:
        result = await session.execute(
            select(DepartmentContent).where(
                and_(
                    DepartmentContent.document_id == doc_uuid,
                    DepartmentContent.department.ilike(department)
                )
            )
        )
        dept_entries = result.scalars().all()
        if not dept_entries:
            raise HTTPException(status_code=404, detail="Department content not found")

        current_state = all([entry.approved for entry in dept_entries])
        new_state = not current_state

        for entry in dept_entries:
            entry.approved = new_state
            session.add(entry)
        await session.commit()

        result_all = await session.execute(
            select(DepartmentContent).where(DepartmentContent.document_id == doc_uuid)
        )
        all_depts = result_all.scalars().all()

        result_doc = await session.execute(select(Document).where(Document.id == doc_uuid))
        doc = result_doc.scalar_one()
        doc.approved = all([d.approved for d in all_depts]) if all_depts else False
        session.add(doc)
        await session.commit()

        return {
            "detail": f"{department} approval toggled to {new_state}",
            "department_approved": new_state,
            "document_approved": doc.approved
        }

# ------------------ Get Department Paragraphs (Only Pages with Content) ------------------ #
@router.get("/api/v1/departments/{doc_id}/{department}")
async def get_department_paragraphs(doc_id: str, department: str):
    import uuid
    from fastapi import HTTPException
    from sqlalchemy import select, and_
    from services import db_service
    from models import Document, DepartmentContent

    try:
        doc_uuid = uuid.UUID(doc_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document_id format")

    async with db_service.SessionLocal() as session:
        # Fetch document
        result_doc = await session.execute(
            select(Document).where(Document.id == doc_uuid)
        )
        doc = result_doc.scalar_one_or_none()
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")

        # Fetch department content
        result = await session.execute(
            select(DepartmentContent)
            .where(
                and_(
                    DepartmentContent.document_id == doc_uuid,
                    DepartmentContent.department.ilike(department)
                )
            )
            .order_by(DepartmentContent.page_start)
        )
        dept_entries = result.scalars().all()
        if not dept_entries:
            raise HTTPException(status_code=404, detail="No content found for this department")

        # Organize pages by page_start
        pages_dict = {}
        for entry in dept_entries:
            page_num = entry.page_start
            if page_num not in pages_dict:
                pages_dict[page_num] = {
                    "page_start": entry.page_start,
                    "page_end": entry.page_end,
                    "confidence": entry.confidence,
                    "keywords_matched": entry.keywords_matched,
                    "pdf_path": entry.pdf_path,
                    "original_path": doc.original_path,
                    "priority": doc.priority,
                    "approved": entry.approved,
                    "created_at": doc.created_at.strftime("%Y-%m-%d"),
                    "content_list": []
                }
            pages_dict[page_num]["content_list"].append(entry.content.strip())

        # Only include pages with actual content
        paragraphs = []
        for page_num in sorted(pages_dict.keys()):
            page_data = pages_dict[page_num]
            if not page_data["content_list"] or all([not c for c in page_data["content_list"]]):
                continue  # Skip empty pages

            paragraphs.append({
                "content": "\n\n".join(page_data["content_list"]),
                "page_start": page_data["page_start"],
                "page_end": page_data["page_end"],
                "confidence": page_data["confidence"],
                "keywords_matched": page_data["keywords_matched"],
                "pdf_path": page_data["pdf_path"],
                "original_path": page_data["original_path"],
                "priority": page_data["priority"],
                "approved": page_data["approved"],
                "created_at": page_data["created_at"]
            })

        return paragraphs

# ------------------ Download Department PDF ------------------ #
@router.get("/api/v1/departments/{doc_id}/{department}/download")
async def download_department_pdf(doc_id: str, department: str):
    try:
        doc_uuid = uuid.UUID(doc_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document_id format")

    async with db_service.SessionLocal() as session:
        result = await session.execute(
            select(DepartmentContent).where(
                and_(
                    DepartmentContent.document_id == doc_uuid,
                    DepartmentContent.department.ilike(department)
                )
            )
        )
        dept_entries = result.scalars().all()
        if not dept_entries:
            raise HTTPException(status_code=404, detail="No content found for this department")

        for entry in dept_entries:
            if entry.pdf_path and os.path.exists(entry.pdf_path):
                return FileResponse(
                    entry.pdf_path,
                    media_type="application/pdf",
                    filename=os.path.basename(entry.pdf_path)
                )

        raise HTTPException(status_code=404, detail="PDF not found")

# ------------------ Download Original Document ------------------ #
@router.get("/api/v1/documents/{doc_id}/download-original")
async def download_original_document(doc_id: str):
    try:
        doc_uuid = uuid.UUID(doc_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document_id format")

    async with db_service.SessionLocal() as session:
        result = await session.execute(
            select(Document).where(Document.id == doc_uuid)
        )
        doc = result.scalar_one_or_none()
        if not doc or not doc.original_path or not os.path.exists(doc.original_path):
            raise HTTPException(status_code=404, detail="Original document not found")

        return FileResponse(
            doc.original_path,
            media_type="application/octet-stream",
            filename=os.path.basename(doc.original_path)
        )

# ------------------ Get Page-wise OCR Text with Tables ------------------ #
@router.get("/api/v1/documents/{doc_id}/ocr-text")
async def get_pagewise_ocr_text(doc_id: str):
    try:
        doc_uuid = uuid.UUID(doc_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document_id format")

    async with db_service.SessionLocal() as session:
        result_doc = await session.execute(
            select(Document).where(Document.id == doc_uuid)
        )
        doc = result_doc.scalar_one_or_none()
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")

        try:
            ocr_pages = ocr_service.extract_text_and_tables(doc.file_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"OCR extraction failed: {e}")

        pages = []
        for page in ocr_pages:
            pages.append({
                "page_number": page["page_number"],
                "text": page["text"],
                "tables": page["tables"]
            })

        return {
            "document_id": str(doc.id),
            "filename": doc.filename,
            "pages": pages
        }
        
# ------------------ Department Stats ------------------ #
@router.get("/api/v1/departments/{department}/stats")
async def get_department_stats(department: str):
    async with db_service.SessionLocal() as session:
        result = await session.execute(
            select(DepartmentContent)
            .where(DepartmentContent.department.ilike(department))
        )
        dept_entries = result.scalars().all()

        total = len(dept_entries)
        high = len([d for d in dept_entries if d.doc_priority == "High"])
        low = len([d for d in dept_entries if d.doc_priority == "Low"])
        approved = len([d for d in dept_entries if d.approved])
        pending = total - approved

        return {
            "department": department,
            "total": total,
            "high": high,
            "low": low,
            "approved": approved,
            "pending": pending
        }
