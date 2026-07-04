from __future__ import annotations

from datetime import datetime
from threading import Lock
from typing import Any


class SessionStore:
    """Small in-memory store that is intentionally reset on server restart."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._history: list[dict[str, Any]] = []
        self._temporary_context = ""
        self._temporary_documents: list[dict[str, Any]] = []
        self._uploaded_files: list[dict[str, Any]] = []
        self._auto_pdf_questions: list[dict[str, Any]] = []
        self._auto_pdf_uploads: list[dict[str, Any]] = []
        self._auto_pdf_metadata: dict[str, Any] = {}

    def add_chat(
        self,
        question: str,
        answers: dict[str, Any],
        *,
        source: str = "manual_chat",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        entry = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "question": question,
            "answers": answers,
            "source": source,
        }
        if metadata:
            entry["metadata"] = metadata
            if metadata.get("question_type"):
                entry["question_type"] = metadata["question_type"]
            if metadata.get("category"):
                entry["category"] = metadata["category"]
        with self._lock:
            self._history.append(entry)
        return entry

    def get_history(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._history)

    def get_temporary_context(self) -> str:
        with self._lock:
            return self._temporary_context

    def get_temporary_documents(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(document) for document in self._temporary_documents]

    def temporary_files_count(self) -> int:
        with self._lock:
            return len(self._temporary_documents)

    def add_temporary_context(
        self,
        filename: str,
        extracted_text: str,
        *,
        max_chars: int = 20000,
    ) -> str:
        clean_text = " ".join((extracted_text or "").split())
        with self._lock:
            self._temporary_documents.append({
                "filename": filename,
                "text": clean_text,
                "characters": len(clean_text),
            })
            combined = "\n\n".join(
                part
                for part in [self._temporary_context, f"[{filename}]\n{clean_text}"]
                if part
            )
            self._temporary_context = combined[:max_chars]
            self._uploaded_files.append({
                "filename": filename,
                "characters": len(clean_text),
                "stored_characters": len(self._temporary_context),
            })
            return self._temporary_context

    def get_uploaded_files(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._uploaded_files)

    def set_auto_pdf_questions(
        self,
        questions: list[dict[str, Any]],
        *,
        filename: str,
        stored_filename: str,
        stored_path: str,
        extraction_metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        with self._lock:
            self._auto_pdf_questions = [dict(question) for question in questions]
            self._auto_pdf_metadata = dict(extraction_metadata or {})
            self._auto_pdf_uploads.append({
                "filename": filename,
                "stored_filename": stored_filename,
                "stored_path": stored_path,
                "question_count": len(questions),
                "extraction_metadata": dict(self._auto_pdf_metadata),
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            })
            return [dict(question) for question in self._auto_pdf_questions]

    def get_auto_pdf_questions(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(question) for question in self._auto_pdf_questions]

    def get_auto_pdf_metadata(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._auto_pdf_metadata)

    def get_auto_pdf_upload_paths(self) -> list[str]:
        with self._lock:
            return [
                str(upload.get("stored_path", ""))
                for upload in self._auto_pdf_uploads
                if upload.get("stored_path")
            ]

    def clear_auto_pdf_questions(self) -> list[str]:
        with self._lock:
            paths = [
                str(upload.get("stored_path", ""))
                for upload in self._auto_pdf_uploads
                if upload.get("stored_path")
            ]
            self._auto_pdf_questions.clear()
            self._auto_pdf_uploads.clear()
            self._auto_pdf_metadata.clear()
            return paths

    def clear_live_state(self, *, keep_temporary_rag: bool = False) -> None:
        """Reset live evaluation state without dropping loaded question sets."""
        with self._lock:
            self._history.clear()
            if not keep_temporary_rag:
                self._temporary_context = ""
                self._temporary_documents.clear()
                self._uploaded_files.clear()

    def clear(self) -> None:
        with self._lock:
            self._history.clear()
            self._temporary_context = ""
            self._temporary_documents.clear()
            self._uploaded_files.clear()
            self._auto_pdf_questions.clear()
            self._auto_pdf_uploads.clear()
            self._auto_pdf_metadata.clear()


session_store = SessionStore()
