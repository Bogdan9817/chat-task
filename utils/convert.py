import os
import io

from langchain_core.documents import Document
from fastapi import UploadFile
from pypdf import PdfReader


def get_file_ext(file: UploadFile):
    filename = file.filename.lower()
    return os.path.splitext(filename)[1]


async def convert_files_to_document(files: list[UploadFile]) -> list[Document]:
    documents = []
    for file in files:
        ext = get_file_ext(file)
        if ext in [".md", ".txt"] or file.content_type in [
            "text/markdown",
            "text/x-markdown",
            "text/plain",
        ]:
            content = (await file.read()).decode("utf-8")
            doc = Document(
                page_content=content,
                metadata={"filename": file.filename, "content_type": file.content_type},
            )
            documents.append(doc)
        elif ext == ".pdf" or file.content_type == "application/pdf":
            content = await file.read()
            reader = PdfReader(io.BytesIO(content))
            pdf_text = "\n".join([page.extract_text() or "" for page in reader.pages])
            documents.append(
                Document(page_content=pdf_text, metadata={"filename": file.filename})
            )
        else:
            continue

    return documents
