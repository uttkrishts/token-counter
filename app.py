#!/usr/bin/env python3
"""Lightweight browser token counter for OpenAI, Gemini, and Claude estimates."""

from __future__ import annotations

import base64
import json
import mimetypes
import math
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

HOST = "127.0.0.1"
PORT = 8765
ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"
MAX_BODY_BYTES = 80 * 1024 * 1024
MAX_FILE_BYTES = 50 * 1024 * 1024
PDF_RENDER_DPI = 96
DEFAULT_PDF_PAGE_WIDTH_PX = 816
DEFAULT_PDF_PAGE_HEIGHT_PX = 1056

GEMINI_MODELS = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-pro",
    "gemini-1.5-flash",
    "gemini-1.0-pro",
    "gemini-2.5-pro-preview-06-05",
    "gemini-2.5-pro-preview-05-06",
    "gemini-2.5-pro-exp-03-25",
    "gemini-live-2.5-flash",
    "gemini-2.5-flash-preview-05-20",
    "gemini-2.5-flash-preview-04-17",
    "gemini-2.5-flash-lite-preview-06-17",
    "gemini-2.0-flash-001",
    "gemini-2.0-flash-lite-001",
    "gemini-1.5-pro-002",
    "gemini-1.5-pro-001",
    "gemini-1.5-flash-002",
    "gemini-1.5-flash-001",
    "gemini-1.0-pro-002",
    "gemini-1.0-pro-001",
]

CLAUDE_ESTIMATES = [
    {
        "name": "claude-cl100k",
        "label": "Claude estimate: cl100k_base",
        "method": "cl100k_base proxy",
    },
    {
        "name": "claude-cl100k-1.3",
        "label": "Claude estimate: cl100k_base x 1.3",
        "method": "cl100k_base proxy with 30% uplift",
    },
    {
        "name": "claude-chars-4",
        "label": "Claude estimate: characters / 4",
        "method": "character heuristic",
    },
    {
        "name": "claude-chars-3.5",
        "label": "Claude estimate: characters / 3.5",
        "method": "conservative character heuristic",
    },
]

GEMINI_TOKENIZERS: dict[str, Any] = {}
OCR_ENGINE: Any | None = None

OPENAI_IMAGE_COSTS = [
    ("gpt-4o-mini", 2833, 5667),
    ("gpt-4.1", 85, 170),
    ("gpt-4o", 85, 170),
    ("gpt-5.1", 70, 140),
    ("gpt-5", 70, 140),
    ("o1", 75, 150),
    ("o3", 75, 150),
]


@dataclass(frozen=True)
class TiktokenState:
    module: Any | None
    model_module: Any | None
    error: str | None = None


@dataclass(frozen=True)
class GeminiState:
    local_tokenizer: Any | None
    error: str | None = None


def load_tiktoken() -> TiktokenState:
    try:
        import tiktoken
        import tiktoken.model as model
    except Exception as exc:  # pragma: no cover - environment dependent
        return TiktokenState(None, None, str(exc))

    return TiktokenState(tiktoken, model)


def load_gemini() -> GeminiState:
    try:
        from google.genai import local_tokenizer
    except Exception as exc:  # pragma: no cover - environment dependent
        return GeminiState(None, str(exc))

    return GeminiState(local_tokenizer)


def encoding_for_choice(state: TiktokenState, choice: str, choice_type: str):
    if state.module is None:
        raise RuntimeError(f"tiktoken is not available: {state.error}")

    if choice_type == "encoding":
        return state.module.get_encoding(choice)

    return state.module.encoding_for_model(choice)


def token_count(state: TiktokenState, choice: str, choice_type: str, text: str) -> int:
    encoding = encoding_for_choice(state, choice, choice_type)
    return len(encoding.encode(text))


def gemini_token_count(state: GeminiState, model_name: str, text: str) -> int:
    if state.local_tokenizer is None:
        raise RuntimeError(f"google-genai local tokenizer is not available: {state.error}")

    if model_name not in GEMINI_TOKENIZERS:
        GEMINI_TOKENIZERS[model_name] = state.local_tokenizer.LocalTokenizer(
            model_name=model_name
        )

    result = GEMINI_TOKENIZERS[model_name].count_tokens(text)
    if isinstance(result, int):
        return result
    if isinstance(result, dict) and "total_tokens" in result:
        return int(result["total_tokens"])
    if hasattr(result, "total_tokens"):
        return int(result.total_tokens)
    if hasattr(result, "totalTokens"):
        return int(result.totalTokens)

    raise RuntimeError("Gemini tokenizer returned an unknown response shape.")


def ceil_count(value: float) -> int:
    return int(math.ceil(value)) if value > 0 else 0


def ceil_div(value: int, divisor: int) -> int:
    return int(math.ceil(value / divisor)) if value > 0 else 0


def claude_estimates(state: TiktokenState, text: str) -> dict[str, int]:
    base = token_count(state, "cl100k_base", "encoding", text)
    chars = len(text)
    return {
        "claude-cl100k": base,
        "claude-cl100k-1.3": ceil_count(base * 1.3),
        "claude-chars-4": ceil_count(chars / 4),
        "claude-chars-3.5": ceil_count(chars / 3.5),
    }


def count_text(
    tiktoken_state: TiktokenState,
    gemini_state: GeminiState,
    provider: str,
    choice: str,
    choice_type: str,
    text: str,
) -> dict[str, Any]:
    if provider == "openai":
        return {
            "tokens": token_count(tiktoken_state, choice, choice_type, text),
            "exact": True,
            "method": "tiktoken",
        }

    if provider == "gemini":
        return {
            "tokens": gemini_token_count(gemini_state, choice, text),
            "exact": True,
            "method": "google-genai LocalTokenizer",
        }

    if provider == "claude":
        estimates = claude_estimates(tiktoken_state, text)
        if choice not in estimates:
            raise ValueError("Select a Claude estimate profile.")
        method = next(
            item["method"] for item in CLAUDE_ESTIMATES if item["name"] == choice
        )
        return {
            "tokens": estimates[choice],
            "exact": False,
            "method": method,
            "estimates": estimates,
        }

    raise ValueError("Provider must be OpenAI, Gemini, or Claude.")


def decode_base64_file(name: str, data: str) -> bytes:
    try:
        return base64.b64decode(data, validate=True)
    except Exception as exc:
        raise ValueError(f"{name}: file data must be valid base64.") from exc


def looks_like_pdf(name: str, media_type: str, content: bytes) -> bool:
    return (
        media_type == "application/pdf"
        or name.lower().endswith(".pdf")
        or content.startswith(b"%PDF")
    )


def looks_like_image(name: str, media_type: str) -> bool:
    image_exts = (".png", ".jpg", ".jpeg", ".webp", ".gif")
    return media_type.startswith("image/") or name.lower().endswith(image_exts)


def image_dimensions(content: bytes) -> tuple[int, int]:
    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("Pillow is required for image token estimates.") from exc

    try:
        with Image.open(BytesIO(content)) as image:
            return int(image.width), int(image.height)
    except Exception as exc:
        raise ValueError("Could not read image dimensions.") from exc


def ocr_engine():
    global OCR_ENGINE
    if OCR_ENGINE is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
        except Exception as exc:
            raise RuntimeError(
                "OCR dependencies are not available. Install requirements.txt first."
            ) from exc
        OCR_ENGINE = RapidOCR()

    return OCR_ENGINE


def ocr_image_bytes(content: bytes) -> str:
    result = ocr_engine()(content)
    rows = result[0] if isinstance(result, tuple) else result
    if not rows:
        return ""

    lines = []
    for row in rows:
        if len(row) >= 2 and row[1]:
            lines.append(str(row[1]))
    return "\n".join(lines).strip()


def render_pdf_pages_for_ocr(content: bytes) -> list[bytes]:
    try:
        import pypdfium2 as pdfium
    except Exception as exc:
        raise RuntimeError(
            "pypdfium2 is required to render PDF pages for OCR."
        ) from exc

    pdf = pdfium.PdfDocument(content)
    images = []
    for index in range(len(pdf)):
        page = pdf[index]
        pil_image = page.render(scale=2).to_pil()
        buffer = BytesIO()
        pil_image.save(buffer, format="PNG")
        images.append(buffer.getvalue())
        page.close()
    pdf.close()
    return images


def should_ocr_pdf(mode: str, extracted_text: str, pages: int) -> bool:
    if mode == "always":
        return True
    if mode == "auto":
        return len(extracted_text.strip()) < max(40, pages * 20)
    return False


def pdf_text_and_pages(content: bytes) -> dict[str, Any]:
    try:
        from pypdf import PdfReader
    except Exception as exc:
        raise RuntimeError("pypdf is required for PDF token estimates.") from exc

    try:
        reader = PdfReader(BytesIO(content))
        if reader.is_encrypted:
            reader.decrypt("")
    except Exception as exc:
        raise ValueError("Could not open PDF file.") from exc

    text_parts = []
    page_sizes = []
    for page in reader.pages:
        try:
            text_parts.append(page.extract_text() or "")
        except Exception:
            text_parts.append("")

        try:
            box = page.mediabox
            width_px = ceil_count(float(box.width) / 72 * PDF_RENDER_DPI)
            height_px = ceil_count(float(box.height) / 72 * PDF_RENDER_DPI)
        except Exception:
            width_px = DEFAULT_PDF_PAGE_WIDTH_PX
            height_px = DEFAULT_PDF_PAGE_HEIGHT_PX

        page_sizes.append(
            {
                "width": width_px or DEFAULT_PDF_PAGE_WIDTH_PX,
                "height": height_px or DEFAULT_PDF_PAGE_HEIGHT_PX,
            }
        )

    return {
        "text": "\n\n".join(text_parts).strip(),
        "pages": len(reader.pages),
        "pageSizes": page_sizes,
    }


def openai_image_cost_params(choice: str) -> tuple[int, int]:
    lowered = choice.lower()
    for prefix, base, tile in OPENAI_IMAGE_COSTS:
        if lowered.startswith(prefix):
            return base, tile
    return 85, 170


def openai_effective_detail(choice: str, media_detail: str) -> str:
    if media_detail in {"low", "high"}:
        return media_detail

    lowered = choice.lower()
    if lowered.startswith("gpt-5.6") or lowered.startswith("gpt-6"):
        return "high"
    return "low"


def openai_image_tokens(width: int, height: int, choice: str, media_detail: str) -> int:
    base, tile = openai_image_cost_params(choice)
    if openai_effective_detail(choice, media_detail) == "low":
        return base

    scale = min(1.0, 2048 / max(width, height))
    scaled_w = max(1, int(width * scale))
    scaled_h = max(1, int(height * scale))

    shortest = min(scaled_w, scaled_h)
    if shortest > 768:
        scale = 768 / shortest
        scaled_w = max(1, int(scaled_w * scale))
        scaled_h = max(1, int(scaled_h * scale))

    tiles = ceil_div(scaled_w, 512) * ceil_div(scaled_h, 512)
    return base + (tiles * tile)


def gemini_image_tokens(width: int, height: int) -> int:
    if width <= 384 and height <= 384:
        return 258
    return ceil_div(width, 768) * ceil_div(height, 768) * 258


def claude_image_tokens(width: int, height: int, media_detail: str) -> int:
    long_edge = 2576 if media_detail == "high" else 1568
    token_limit = 4784 if media_detail == "high" else 1568
    scale = min(1.0, long_edge / max(width, height))

    def count_at(current_scale: float) -> tuple[int, int, int]:
        scaled_w = max(1, int(width * current_scale))
        scaled_h = max(1, int(height * current_scale))
        tokens = ceil_div(scaled_w, 28) * ceil_div(scaled_h, 28)
        return scaled_w, scaled_h, tokens

    scaled_w, scaled_h, tokens = count_at(scale)
    while tokens > token_limit or max(scaled_w, scaled_h) > long_edge:
        scale *= 0.98
        scaled_w, scaled_h, tokens = count_at(scale)

    return tokens


def visual_tokens_for_image(
    provider: str,
    choice: str,
    media_detail: str,
    width: int,
    height: int,
) -> dict[str, Any]:
    if provider == "openai":
        detail = openai_effective_detail(choice, media_detail)
        return {
            "tokens": openai_image_tokens(width, height, choice, media_detail),
            "method": f"OpenAI image estimate ({detail} detail)",
        }
    if provider == "gemini":
        return {
            "tokens": gemini_image_tokens(width, height),
            "method": "Gemini image estimate, 258 tokens per image/tile",
        }
    if provider == "claude":
        tier = "high-resolution" if media_detail == "high" else "standard"
        return {
            "tokens": claude_image_tokens(width, height, media_detail),
            "method": f"Claude {tier} visual-token estimate",
        }

    raise ValueError("Provider must be OpenAI, Gemini, or Claude.")


def visual_tokens_for_pdf(
    provider: str,
    choice: str,
    media_detail: str,
    page_sizes: list[dict[str, int]],
) -> dict[str, Any]:
    if provider == "gemini":
        return {
            "tokens": len(page_sizes) * 258,
            "method": "Gemini PDF estimate, 258 tokens per page",
        }

    total = 0
    methods = set()
    for page in page_sizes:
        result = visual_tokens_for_image(
            provider,
            choice,
            media_detail,
            int(page["width"]),
            int(page["height"]),
        )
        total += result["tokens"]
        methods.add(result["method"])

    return {"tokens": total, "method": " + ".join(sorted(methods))}


def json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, indent=2).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def error_response(handler: BaseHTTPRequestHandler, status: int, message: str) -> None:
    json_response(handler, status, {"ok": False, "error": message})


def read_json(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length", "0"))
    if length > MAX_BODY_BYTES:
        raise ValueError("Request is too large.")

    raw = handler.rfile.read(length)
    if not raw:
        return {}

    try:
        return json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("Request body must be valid JSON.") from exc


def model_payload(tiktoken_state: TiktokenState, gemini_state: GeminiState) -> dict[str, Any]:
    openai: dict[str, Any] = {
        "available": tiktoken_state.module is not None and tiktoken_state.model_module is not None,
        "models": [],
        "modelPrefixes": [],
        "encodings": [],
        "error": tiktoken_state.error,
        "install": "python3 -m pip install -r requirements.txt",
    }

    if openai["available"]:
        exact_models = sorted(tiktoken_state.model_module.MODEL_TO_ENCODING)
        model_prefixes = sorted(tiktoken_state.model_module.MODEL_PREFIX_TO_ENCODING)
        openai.update(
            {
                "models": [
                    {
                        "name": name,
                        "encoding": tiktoken_state.model_module.MODEL_TO_ENCODING[name],
                    }
                    for name in exact_models
                ],
                "modelPrefixes": [
                    {
                        "prefix": prefix,
                        "encoding": tiktoken_state.model_module.MODEL_PREFIX_TO_ENCODING[prefix],
                    }
                    for prefix in model_prefixes
                ],
                "encodings": sorted(tiktoken_state.module.list_encoding_names()),
            }
        )

    return {
        "ok": True,
        "providers": {
            "openai": openai,
            "gemini": {
                "available": gemini_state.local_tokenizer is not None,
                "models": [{"name": name} for name in GEMINI_MODELS],
                "error": gemini_state.error,
                "install": "python3 -m pip install -r requirements.txt",
            },
            "claude": {
                "available": openai["available"],
                "estimates": CLAUDE_ESTIMATES,
                "error": tiktoken_state.error,
                "install": "python3 -m pip install -r requirements.txt",
                "note": "Claude counts are estimates only without Anthropic API access.",
            },
        },
    }


def normalize_file(item: dict[str, Any]) -> dict[str, Any]:
    name = str(item.get("name") or "untitled.txt")
    media_type = str(item.get("type") or "").lower()
    content = item.get("content")
    encoded = item.get("data")
    declared_size = int(item.get("size") or 0)

    if isinstance(encoded, str) and encoded:
        raw = decode_base64_file(name, encoded)
    elif isinstance(content, str):
        raw = content.encode("utf-8")
    else:
        raise ValueError(f"{name}: file content is missing.")

    size = len(raw)
    if size > MAX_FILE_BYTES:
        raise ValueError(
            f"{name}: file is larger than {MAX_FILE_BYTES // (1024 * 1024)} MB."
        )
    if declared_size and declared_size != size and not content:
        raise ValueError(f"{name}: uploaded file size did not match transmitted data.")

    if looks_like_pdf(name, media_type, raw):
        return {"name": name, "type": media_type, "kind": "pdf", "bytes": size, "raw": raw}
    if looks_like_image(name, media_type):
        return {"name": name, "type": media_type, "kind": "image", "bytes": size, "raw": raw}

    if isinstance(content, str):
        text = content
    else:
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"{name}: unsupported binary file. Use text, PDF, or image files."
            ) from exc

    return {"name": name, "type": media_type, "kind": "text", "bytes": size, "content": text}


def combined_text(prompt: str, files: list[dict[str, Any]]) -> str:
    combined = prompt
    text_files = [item for item in files if item.get("content")]
    if text_files:
        file_text = "\n\n".join(
            f"--- {item['name']} ---\n{item['content']}" for item in text_files
        )
        combined = f"{prompt}\n\n{file_text}" if prompt else file_text

    return combined


def count_file(
    tiktoken_state: TiktokenState,
    gemini_state: GeminiState,
    provider: str,
    choice: str,
    choice_type: str,
    media_detail: str,
    ocr_mode: str,
    item: dict[str, Any],
) -> dict[str, Any]:
    text = ""
    ocr_text = ""
    media_tokens = 0
    media_method = ""
    dimensions = ""
    pages = 0

    if item["kind"] == "text":
        text = item["content"]
        file_type = "Text"
    elif item["kind"] == "pdf":
        pdf = pdf_text_and_pages(item["raw"])
        text = pdf["text"]
        pages = pdf["pages"]
        visual = visual_tokens_for_pdf(
            provider, choice, media_detail, pdf["pageSizes"]
        )
        media_tokens = visual["tokens"]
        media_method = visual["method"]
        if should_ocr_pdf(ocr_mode, text, pages):
            ocr_text = "\n\n".join(
                filter(None, (ocr_image_bytes(image) for image in render_pdf_pages_for_ocr(item["raw"])))
            ).strip()
            text = "\n\n".join(filter(None, [text, ocr_text]))
        suffix = ", OCR" if ocr_text else ""
        file_type = f"PDF ({pages} pages{suffix})"
    elif item["kind"] == "image":
        width, height = image_dimensions(item["raw"])
        dimensions = f"{width}x{height}"
        visual = visual_tokens_for_image(provider, choice, media_detail, width, height)
        media_tokens = visual["tokens"]
        media_method = visual["method"]
        if ocr_mode != "off":
            ocr_text = ocr_image_bytes(item["raw"])
            text = ocr_text
        suffix = ", OCR" if ocr_text else ""
        file_type = f"Image ({dimensions}{suffix})"
    else:
        raise ValueError(f"{item['name']}: unsupported file type.")

    text_result = count_text(
        tiktoken_state,
        gemini_state,
        provider,
        choice,
        choice_type,
        text,
    )
    text_tokens = text_result["tokens"]
    total_tokens = text_tokens + media_tokens
    method_parts = [text_result["method"]]
    if ocr_text:
        method_parts.append("OCR text")
    if media_method:
        method_parts.append(media_method)

    exact = text_result["exact"] and media_tokens == 0 and not ocr_text
    return {
        "name": item["name"],
        "type": file_type,
        "chars": len(text),
        "bytes": item["bytes"],
        "pages": pages,
        "dimensions": dimensions,
        "tokens": total_tokens,
        "textTokens": text_tokens,
        "mediaTokens": media_tokens,
        "ocrChars": len(ocr_text),
        "exact": exact,
        "method": " + ".join(method_parts),
        "text": text,
    }


def claude_combined_estimates(
    tiktoken_state: TiktokenState,
    text: str,
    media_tokens: int,
) -> dict[str, int]:
    return {
        name: tokens + media_tokens
        for name, tokens in claude_estimates(tiktoken_state, text).items()
    }


def summarize_estimates(estimates: dict[str, int] | None) -> list[dict[str, Any]]:
    if not estimates:
        return []

    labels = {item["name"]: item["label"] for item in CLAUDE_ESTIMATES}
    methods = {item["name"]: item["method"] for item in CLAUDE_ESTIMATES}
    return [
        {
            "name": name,
            "label": labels[name],
            "method": methods[name],
            "tokens": estimates[name],
        }
        for name in labels
    ]


def count_payload(
    tiktoken_state: TiktokenState,
    gemini_state: GeminiState,
    data: dict[str, Any],
) -> dict[str, Any]:
    provider = str(data.get("provider") or "openai").strip()
    choice = str(data.get("choice") or "").strip()
    choice_type = str(data.get("choiceType") or "model").strip()
    media_detail = str(data.get("mediaDetail") or "auto").strip()
    ocr_mode = str(data.get("ocrMode") or "off").strip()
    prompt = data.get("prompt", "")
    raw_files = data.get("files", [])

    if provider not in {"openai", "gemini", "claude"}:
        raise ValueError("Provider must be OpenAI, Gemini, or Claude.")
    if provider == "openai" and choice_type not in {"model", "encoding"}:
        raise ValueError("OpenAI choiceType must be either 'model' or 'encoding'.")
    if media_detail not in {"auto", "low", "high"}:
        raise ValueError("Media detail must be auto, low, or high.")
    if ocr_mode not in {"off", "auto", "always"}:
        raise ValueError("OCR mode must be off, auto, or always.")
    if not choice:
        raise ValueError("Select a tokenizer first.")
    if not isinstance(prompt, str):
        raise ValueError("Prompt must be text.")
    if not isinstance(raw_files, list):
        raise ValueError("Files must be a list.")

    files = [normalize_file(item) for item in raw_files]
    prompt_result = count_text(
        tiktoken_state,
        gemini_state,
        provider,
        choice,
        choice_type,
        prompt,
    )

    file_results = []
    file_tokens = 0
    media_tokens = 0
    text_parts = []
    for item in files:
        result = count_file(
            tiktoken_state,
            gemini_state,
            provider,
            choice,
            choice_type,
            media_detail,
            ocr_mode,
            item,
        )
        count = result["tokens"]
        file_tokens += count
        media_tokens += result["mediaTokens"]
        if result["text"]:
            text_parts.append({"name": item["name"], "content": result["text"]})
        file_results.append({key: value for key, value in result.items() if key != "text"})

    combined_file_text = combined_text(prompt, text_parts)
    if provider == "claude":
        estimate_values = claude_combined_estimates(
            tiktoken_state, combined_file_text, media_tokens
        )
        combined_text_tokens = estimate_values[choice] - media_tokens
        combined_result = {
            "tokens": estimate_values[choice],
            "exact": False,
            "method": " + ".join(
                part
                for part in [
                    next(
                        item["method"]
                        for item in CLAUDE_ESTIMATES
                        if item["name"] == choice
                    ),
                    "multimodal estimate" if media_tokens else "",
                ]
                if part
            ),
            "estimates": estimate_values,
        }
    else:
        combined_text_result = count_text(
            tiktoken_state,
            gemini_state,
            provider,
            choice,
            choice_type,
            combined_file_text,
        )
        combined_text_tokens = combined_text_result["tokens"]
        combined_result = {
            "tokens": combined_text_tokens + media_tokens,
            "exact": combined_text_result["exact"] and media_tokens == 0,
            "method": combined_text_result["method"]
            + (" + multimodal estimate" if media_tokens else ""),
        }

    return {
        "ok": True,
        "provider": provider,
        "choice": choice,
        "choiceType": choice_type,
        "mediaDetail": media_detail,
        "ocrMode": ocr_mode,
        "exact": combined_result["exact"],
        "method": combined_result["method"],
        "prompt": {
            "chars": len(prompt),
            "bytes": len(prompt.encode("utf-8")),
            "tokens": prompt_result["tokens"],
            "exact": prompt_result["exact"],
            "method": prompt_result["method"],
        },
        "files": file_results,
        "fileTokens": file_tokens,
        "mediaTokens": media_tokens,
        "combinedTextTokens": combined_text_tokens,
        "combinedTokens": combined_result["tokens"],
        "estimates": summarize_estimates(combined_result.get("estimates")),
    }


class TokenCounterHandler(BaseHTTPRequestHandler):
    server_version = "TokenCounter/1.0"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/models":
            json_response(
                self,
                HTTPStatus.OK,
                model_payload(load_tiktoken(), load_gemini()),
            )
            return

        if parsed.path == "/":
            self.serve_static("index.html")
            return

        static_path = parsed.path.removeprefix("/")
        self.serve_static(static_path)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/count":
            error_response(self, HTTPStatus.NOT_FOUND, "Endpoint not found.")
            return

        try:
            payload = count_payload(load_tiktoken(), load_gemini(), read_json(self))
        except ValueError as exc:
            error_response(self, HTTPStatus.BAD_REQUEST, str(exc))
            return
        except Exception as exc:
            error_response(self, HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))
            return

        json_response(self, HTTPStatus.OK, payload)

    def serve_static(self, relative_path: str) -> None:
        target = (STATIC_DIR / relative_path).resolve()
        if not str(target).startswith(str(STATIC_DIR.resolve())) or not target.is_file():
            error_response(self, HTTPStatus.NOT_FOUND, "File not found.")
            return

        content_type, _ = mimetypes.guess_type(target)
        body = target.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format % args}")


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), TokenCounterHandler)
    print(f"Token counter running at http://{HOST}:{PORT}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping token counter.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
