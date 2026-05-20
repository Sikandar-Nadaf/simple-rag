from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown"}


@dataclass(slots=True)
class Document:
    source: str
    text: str


def load_documents(data_dir: Path) -> list[Document]:
    documents: list[Document] = []
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        text = path.read_text(encoding="utf-8")
        if text.strip():
            documents.append(Document(source=str(path.relative_to(data_dir)), text=text))
    return documents
