from pathlib import Path
from langchain_core.documents import Document


def txt_folder_to_documents(folder_path: str) -> list[Document]:
    documents = []
    folder = Path(folder_path)
    txt_files = folder.glob("*.txt")

    for file_path in txt_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            doc = Document(
                page_content=content,
                metadata={"source": str(file_path), "filename": file_path.name},
            )
            documents.append(doc)

        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    return documents
