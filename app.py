import io
import os
import tempfile
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from pydantic import BaseModel
from pypdf import PdfReader, PdfWriter
import indra_ocr
from typesafe import TypeSafeClient
from typesafe.types import Choice

app = FastAPI(title="Insurance Underwriting Document Classifier")

# ---------------------------------------------------------
# Document Choices Configuration
# ---------------------------------------------------------
INSURANCE_DOC_CHOICES = {
    "Loss Run": (
        "A carrier issued report detailing an insured's historical claims, "
        "paid amounts, and outstanding reserves over time."
    ),
    "ACORD": (
        "A standardized industry form used by brokers to formally request coverage "
        "and summarize an applicant's operations and risks."
    ),
    "SOV": (
        "An itemized schedule of an insured's physical locations and property values "
        "used for property rating."
    ),
    "Market Presentation": (
        "A broker prepared narrative deck pitching the insured's business profile, "
        "risk management practices, and coverage goals to prospective underwriters."
    ),
    "Other": (
        "Any other document that cannot be reliably defined as the above mentioned."
    ),
}

# Pre-load Indra OCR engine on startup
conv = indra_ocr.load()


class ClassificationResult(BaseModel):
    filename: str
    doc_type: str
    extracted_preview: str


def extract_first_page_pdf(pdf_bytes: bytes) -> bytes:
    """Isolates page 1 of the uploaded PDF in memory."""
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        if len(reader.pages) == 0:
            raise ValueError("The PDF contains no pages.")

        writer = PdfWriter()
        writer.add_page(reader.pages[0])

        out = io.BytesIO()
        writer.write(out)
        return out.getvalue()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid or corrupted PDF: {str(e)}",
        )


def run_indra_first_page(first_page_bytes: bytes) -> str:
    """Runs Indra OCR on the first page bytes."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
        tmp.write(first_page_bytes)
        tmp.flush()

        extracted_text = ""
        for _, page in conv.convert(tmp.name):
            extracted_text = getattr(page, "text", "") or getattr(page, "markdown", "")
            break
        return extracted_text


@app.post("/classify", response_model=ClassificationResult)
async def classify_underwriting_doc(file: UploadFile = File(...)):
    # 1. Enforce PDF only
    if not file.filename.lower().endswith(".pdf") or file.content_type != "application/pdf":
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported file format. Please upload a PDF document (*.pdf).",
        )

    # 2. Extract only page 1
    file_bytes = await file.read()
    page1_bytes = extract_first_page_pdf(file_bytes)

    # 3. OCR first page with Indra
    page_text = run_indra_first_page(page1_bytes)
    if not page_text.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No readable text detected on the first page.",
        )

    # 4. Classify via TypeSafe
    try:
        with TypeSafeClient() as client:
            response = client.system_one(
                state=page_text,
                questions={
                    "doc_type": Choice(
                        instructions="What is the document type for insurance underwriting?",
                        criteria=INSURANCE_DOC_CHOICES,
                    )
                },
            )

        predicted_choice = response.choices.get("doc_type")
        choice_val = getattr(predicted_choice, "value", str(predicted_choice))

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"TypeSafe classification error: {str(e)}",
        )

    return ClassificationResult(
        filename=file.filename,
        doc_type=choice_val,
        extracted_preview=page_text[:300].strip(),
    )