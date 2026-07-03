from __future__ import annotations

from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.cleanup import cleanup_temp_uploads, register_cleanup
from backend.evaluator_bridge import bridge_status, evaluate_answer, mark_demo_evaluation
from backend.metrics import build_csv, build_pdf, compute_session_metrics
from backend.pdf_auto_questions import extract_questions_from_pdf_result
from backend.rag_temp import main_project_rag_status
from backend.rag_temp import save_and_extract_upload
from backend.rag_temp import safe_filename
from backend.runtime_bridge import answer_question, health_status
from backend.session_store import session_store


BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"
TEMP_UPLOAD_DIR = BASE_DIR / "temp_uploads"

app = FastAPI(title="Voice Assistant Website", version="1.0.0")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
register_cleanup(TEMP_UPLOAD_DIR)


class ChatRequest(BaseModel):
    question: str
    system: str = "C"
    research_evaluation_mode: bool = False


class AutoPdfRunRequest(BaseModel):
    system: str = "C"
    count: int | str | None = None
    position: str = "start"
    research_evaluation_mode: bool = False


class AutoPdfRunOneRequest(BaseModel):
    question: str
    system: str = "C"
    source: str = "auto_pdf"
    display_question: str | None = None
    pdf_question_number: int | None = None
    pdf_category: str | None = None
    research_evaluation_mode: bool = False


def _delete_auto_pdf_uploads(paths: list[str]) -> None:
    upload_root = TEMP_UPLOAD_DIR.resolve()
    for raw_path in paths:
        try:
            path = Path(raw_path).resolve()
        except (OSError, RuntimeError):
            continue
        if not path.is_file():
            continue
        try:
            path.relative_to(upload_root)
        except ValueError:
            continue
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _select_question_range(
    questions: list[dict],
    count_input: int | str | None,
    position: str,
) -> tuple[list[dict], int, int, int, int]:
    total = len(questions)
    if total <= 0:
        return [], 0, 0, 0, 0

    count_text = "" if count_input is None else str(count_input).strip()
    if not count_text or count_text.lower() == "all":
        count = total
    else:
        try:
            count = int(count_text)
        except (TypeError, ValueError):
            count = total
        if count <= 0 or count > total:
            count = total

    start_index = 0
    normalized_position = position.strip().lower()
    if count < total:
        if normalized_position == "middle":
            start_index = max(0, (total - count) // 2)
        elif normalized_position == "end":
            start_index = max(0, total - count)

    end_index = min(total, start_index + count)
    return questions[start_index:end_index], start_index, end_index, count, total


def _entry_question(entry: dict | str) -> str:
    if isinstance(entry, dict):
        return str(entry.get("question") or entry.get("raw") or "").strip()
    return str(entry or "").strip()


def _entry_display(entry: dict | str) -> str:
    if isinstance(entry, dict):
        return str(entry.get("raw") or entry.get("question") or "").strip()
    return str(entry or "").strip()


def _entry_number(entry: dict | str) -> int | None:
    if not isinstance(entry, dict):
        return None
    value = entry.get("number")
    return value if isinstance(value, int) else None


def _entry_category(entry: dict | str) -> str | None:
    if not isinstance(entry, dict):
        return None
    value = entry.get("category")
    return str(value) if value else None


def _auto_pdf_response_payload(questions: list[dict]) -> dict:
    metadata = session_store.get_auto_pdf_metadata()
    return {
        "total_questions": len(questions),
        "questions": questions,
        "preview": questions[:10],
        "numbered_entries_found": metadata.get("numbered_entries_found", 0),
        "missing_numbers": metadata.get("missing_numbers", []),
        "duplicate_numbers": metadata.get("duplicate_numbers", []),
        "category_counts": metadata.get("category_counts", {}),
        "extraction_mode": metadata.get("extraction_mode"),
        "fallback_used": metadata.get("fallback_used", False),
    }


def _error_answers(system_choice: str, message: str) -> dict[str, dict]:
    selected = system_choice.upper()
    system_keys = ["A", "B", "C"] if selected == "ALL" else [selected]
    return {
        system_key: {
            "response": message,
            "latency": 0.0,
            "metadata": {
                "error": True,
                "message": message,
                "main_rag_used": False,
                "temporary_rag_used": False,
                "temporary_files_count": session_store.temporary_files_count(),
                "combined_context_chars": 0,
            },
        }
        for system_key in system_keys
    }


def _normalize_system_choice(system: str) -> str:
    selected = system.strip().upper()
    if selected == "ALL":
        return "ALL"
    if selected in {"A", "B", "C"}:
        return selected
    raise HTTPException(status_code=400, detail="System must be A, B, C, or all.")


def _merge_evaluation(answer: dict, evaluation: dict) -> None:
    metadata = answer.setdefault("metadata", {})
    metadata.update(evaluation)
    answer.update(evaluation)


def _apply_evaluation_mode(
    answers: dict[str, dict],
    question: str,
    *,
    research_evaluation_mode: bool,
    metadata: dict | None = None,
) -> dict[str, dict]:
    metadata = dict(metadata or {})
    question_type = (
        metadata.get("question_type")
        or metadata.get("pdf_category")
        or metadata.get("category")
    )
    expected_intent = metadata.get("expected_intent") or question_type
    evaluation_mode = "research" if research_evaluation_mode else "demo"

    for answer in answers.values():
        if not isinstance(answer, dict):
            continue
        answer_metadata = answer.setdefault("metadata", {})
        answer_metadata.update({
            key: value
            for key, value in metadata.items()
            if value is not None
        })
        answer_metadata["evaluation_mode"] = evaluation_mode
        answer["evaluation_mode"] = evaluation_mode

        if research_evaluation_mode:
            evaluation = evaluate_answer(
                question,
                str(answer.get("response") or ""),
                metadata=answer_metadata,
                question_type=question_type,
                expected_intent=expected_intent,
            )
            _merge_evaluation(answer, evaluation)
        else:
            if question_type and question_type != "unknown":
                answer_metadata.setdefault("question_type", question_type)
                answer.setdefault("question_type", question_type)
                answer_metadata.setdefault("expected_intent", expected_intent)
                answer.setdefault("expected_intent", expected_intent)
            mark_demo_evaluation(answer)
    return answers


@app.on_event("startup")
async def startup_cleanup() -> None:
    session_store.clear()
    cleanup_temp_uploads(TEMP_UPLOAD_DIR)


@app.on_event("shutdown")
async def shutdown_cleanup() -> None:
    session_store.clear()
    cleanup_temp_uploads(TEMP_UPLOAD_DIR)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/health")
async def health(research_mode_enabled: bool = False) -> dict:
    health = health_status()
    history = session_store.get_history()
    metrics = compute_session_metrics(history)
    rag_status = main_project_rag_status(validate_import=True)
    bridge = bridge_status()
    temp_files_count = session_store.temporary_files_count()
    health.update({
        **bridge,
        "browser_voice_note": (
            "Mic works best in Chrome/Edge. Firefox may not support Web Speech Recognition."
        ),
        "session_question_count": len(history),
        "session_response_count": metrics["total_system_responses"],
        "systems_available": {
            "A": health.get("system_a_available", False),
            "B": health.get("system_b_available", False),
            "C": health.get("system_c_available", False),
        },
        "main_rag_available": rag_status["main_rag_available"],
        "main_rag_source": rag_status["main_rag_source"],
        "main_rag_path_detected": rag_status["main_rag_path_detected"],
        "main_rag_module": rag_status["main_rag_module"],
        "main_rag_read_only": rag_status["main_rag_read_only"],
        "main_rag_test_query_result_count": rag_status["main_rag_test_query_result_count"],
        "main_rag_error": rag_status["main_rag_error"],
        "temp_uploads_path": str(TEMP_UPLOAD_DIR),
        "temp_files_count": temp_files_count,
        "temp_rag_active": temp_files_count > 0,
        "temporary_rag_active": temp_files_count > 0,
        "research_evaluation_supported": bridge.get("research_evaluation_supported", False),
        "evaluator_bridge_available": bridge.get("evaluator_bridge_available", False),
        "research_evaluation_method": (
            bridge.get("research_evaluation_method")
            if research_mode_enabled
            else bridge.get("demo_evaluation_method", "demo_metrics")
        ),
        "research_mode_currently_enabled": research_mode_enabled,
        "note": "Main project RAG is read-only. Temporary website RAG is session-only.",
    })
    return health


@app.post("/api/chat")
async def chat(request: ChatRequest) -> dict:
    question = request.question.strip()
    system = request.system.strip().upper()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if system not in {"A", "B", "C", "ALL"}:
        raise HTTPException(status_code=400, detail="System must be A, B, C, or all.")

    answers = answer_question(
        question,
        system,
        temporary_context=session_store.get_temporary_context(),
    )
    session_metadata = {
        "evaluation_mode": "research" if request.research_evaluation_mode else "demo",
    }
    _apply_evaluation_mode(
        answers,
        question,
        research_evaluation_mode=request.research_evaluation_mode,
        metadata=session_metadata,
    )
    session_store.add_chat(question, answers, metadata=session_metadata)
    return {
        "question": question,
        "research_evaluation_mode": request.research_evaluation_mode,
        "answers": answers,
    }


@app.post("/api/auto_pdf_extract")
async def auto_pdf_extract(file: UploadFile = File(...)) -> dict:
    original_name = file.filename or "questions.pdf"
    suffix = Path(original_name).suffix.lower()
    if suffix != ".pdf":
        raise HTTPException(status_code=400, detail="Upload a PDF file for auto-question extraction.")

    TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    stored_name = f"auto_pdf_{safe_filename(original_name)}"
    stored_path = TEMP_UPLOAD_DIR / stored_name
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded PDF is empty.")

    stored_path.write_bytes(data)
    try:
        extraction_result = extract_questions_from_pdf_result(stored_path)
        questions = extraction_result["questions"]
    except Exception as exc:
        try:
            stored_path.unlink()
        except FileNotFoundError:
            pass
        raise HTTPException(
            status_code=500,
            detail=f"Could not extract questions from PDF: {type(exc).__name__}: {exc}",
        ) from exc

    stored_questions = session_store.set_auto_pdf_questions(
        questions,
        filename=original_name,
        stored_filename=stored_name,
        stored_path=str(stored_path),
        extraction_metadata={
            key: value
            for key, value in extraction_result.items()
            if key != "questions"
        },
    )
    return _auto_pdf_response_payload(stored_questions)


@app.post("/api/auto_pdf_run")
async def auto_pdf_run(request: AutoPdfRunRequest) -> dict:
    system_for_runtime = _normalize_system_choice(request.system)

    position = request.position.strip().lower()
    if position not in {"start", "middle", "end"}:
        position = "start"

    questions = session_store.get_auto_pdf_questions()
    if not questions:
        raise HTTPException(status_code=400, detail="Extract PDF questions before running.")

    selected_questions, start_index, end_index, selected_count, total_questions = _select_question_range(
        questions,
        request.count,
        position,
    )
    results = []
    for offset, entry in enumerate(selected_questions, start=start_index):
        question = _entry_question(entry)
        display_question = _entry_display(entry)
        pdf_question_number = _entry_number(entry) or offset + 1
        pdf_category = _entry_category(entry)
        try:
            answers = answer_question(question, system_for_runtime)
        except Exception as exc:
            message = f"Auto PDF question failed: {type(exc).__name__}: {exc}"
            answers = _error_answers(system_for_runtime, message)

        metadata = {
            "auto_pdf": {
                "question_index": pdf_question_number,
                "requested_position": position,
            },
            "pdf_question_number": pdf_question_number,
            "pdf_category": pdf_category,
            "question_type": pdf_category,
            "category": pdf_category,
            "expected_intent": pdf_category,
            "display_question": display_question,
            "evaluation_mode": "research" if request.research_evaluation_mode else "demo",
        }
        _apply_evaluation_mode(
            answers,
            question,
            research_evaluation_mode=request.research_evaluation_mode,
            metadata=metadata,
        )
        session_store.add_chat(
            question,
            answers,
            source="auto_pdf",
            metadata=metadata,
        )
        results.append({
            "question": question,
            "display_question": display_question,
            "pdf_question_number": pdf_question_number,
            "pdf_category": pdf_category,
            "question_index": pdf_question_number,
            "answers": answers,
            "source": "auto_pdf",
        })

    return {
        "selected_count": len(selected_questions),
        "requested_count": selected_count,
        "total_questions": total_questions,
        "start_index": start_index,
        "end_index": end_index,
        "results": results,
    }


@app.post("/api/auto_pdf_run_one")
async def auto_pdf_run_one(request: AutoPdfRunOneRequest) -> dict:
    question = request.question.strip()
    source = request.source.strip().lower()
    system_for_runtime = _normalize_system_choice(request.system)

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if source != "auto_pdf":
        raise HTTPException(status_code=400, detail="Source must be auto_pdf.")

    questions = session_store.get_auto_pdf_questions()
    matched_entry = None
    if request.pdf_question_number is not None:
        matched_entry = next(
            (
                entry
                for entry in questions
                if _entry_number(entry) == request.pdf_question_number
            ),
            None,
        )
    if matched_entry is None:
        matched_entry = next(
            (
                entry
                for entry in questions
                if _entry_question(entry) == question
            ),
            None,
        )

    question_index = (
        request.pdf_question_number
        or _entry_number(matched_entry)
        or (questions.index(matched_entry) + 1 if matched_entry in questions else None)
    )
    display_question = (
        request.display_question
        or _entry_display(matched_entry)
        or question
    )
    pdf_category = request.pdf_category or _entry_category(matched_entry)
    try:
        answers = answer_question(
            question,
            system_for_runtime,
            temporary_context=session_store.get_temporary_context(),
        )
    except Exception as exc:
        message = f"Auto PDF question failed: {type(exc).__name__}: {exc}"
        answers = _error_answers(system_for_runtime, message)

    metadata = {"auto_pdf": {}}
    if question_index is not None:
        metadata["auto_pdf"]["question_index"] = question_index
        metadata["pdf_question_number"] = question_index
    if pdf_category:
        metadata["pdf_category"] = pdf_category
        metadata["question_type"] = pdf_category
        metadata["category"] = pdf_category
        metadata["expected_intent"] = pdf_category
    if display_question:
        metadata["display_question"] = display_question
    metadata["evaluation_mode"] = "research" if request.research_evaluation_mode else "demo"
    _apply_evaluation_mode(
        answers,
        question,
        research_evaluation_mode=request.research_evaluation_mode,
        metadata=metadata,
    )

    session_store.add_chat(
        question,
        answers,
        source="auto_pdf",
        metadata=metadata,
    )
    return {
        "question": question,
        "display_question": display_question,
        "pdf_question_number": question_index,
        "pdf_category": pdf_category,
        "source": "auto_pdf",
        "research_evaluation_mode": request.research_evaluation_mode,
        "answers": answers,
    }


@app.get("/api/auto_pdf_questions")
async def auto_pdf_questions() -> dict:
    questions = session_store.get_auto_pdf_questions()
    return _auto_pdf_response_payload(questions)


@app.get("/api/auto_pdf_debug")
async def auto_pdf_debug() -> dict:
    questions = session_store.get_auto_pdf_questions()
    midpoint = max((len(questions) - 5) // 2, 0)
    metadata = session_store.get_auto_pdf_metadata()
    return {
        "extracted_count": len(questions),
        "first_5_questions": questions[:5],
        "middle_5_questions": questions[midpoint:midpoint + 5],
        "last_5_questions": questions[-5:] if questions else [],
        "category_counts": metadata.get("category_counts", {}),
        "numbered_entries_found": metadata.get("numbered_entries_found", 0),
        "missing_numbers": metadata.get("missing_numbers", []),
        "duplicate_numbers": metadata.get("duplicate_numbers", []),
        "extraction_mode": metadata.get("extraction_mode"),
        "fallback_used": metadata.get("fallback_used", False),
    }


@app.post("/api/auto_pdf_clear")
async def auto_pdf_clear() -> dict:
    paths = session_store.clear_auto_pdf_questions()
    _delete_auto_pdf_uploads(paths)
    return {
        "ok": True,
        "message": "Extracted PDF questions cleared.",
        "total_questions": 0,
        "questions": [],
        "preview": [],
        "numbered_entries_found": 0,
        "missing_numbers": [],
        "duplicate_numbers": [],
        "category_counts": {},
    }


@app.get("/api/session_metrics")
async def session_metrics() -> dict:
    return compute_session_metrics(session_store.get_history())


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)) -> dict:
    try:
        upload = await save_and_extract_upload(file, TEMP_UPLOAD_DIR)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not process uploaded file: {type(exc).__name__}: {exc}",
        ) from exc

    context = session_store.add_temporary_context(
        upload["original_filename"],
        upload["text"],
    )
    return {
        "filename": upload["original_filename"],
        "stored_filename": upload["stored_filename"],
        "characters_extracted": upload["characters"],
        "temporary_context_characters": len(context),
        "temp_files_count": session_store.temporary_files_count(),
        "temporary_rag_used": True,
        "preview": context[:500],
    }


@app.post("/api/clear_session")
async def clear_session() -> dict:
    session_store.clear()
    cleanup_temp_uploads(TEMP_UPLOAD_DIR)
    return {"ok": True, "message": "Session, metrics, temporary RAG, and uploads cleared."}


@app.get("/api/export_csv")
async def export_csv() -> Response:
    csv_text = build_csv(session_store.get_history())
    return Response(
        csv_text,
        media_type="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=voice_assistant_session_results.csv"
        },
    )


@app.get("/api/download_pdf")
async def download_pdf() -> Response:
    history = session_store.get_history()
    metrics = compute_session_metrics(history)
    pdf_bytes = build_pdf(history, metrics)
    return Response(
        pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": "attachment; filename=voice_assistant_session_report.pdf"
        },
    )


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )
