# Token Counter

A lightweight browser tool for counting prompt and file tokens.

## Run

```bash
python3 -m pip install -r requirements.txt
python3 app.py
```

Open <http://127.0.0.1:8765>.

## Features

- Lists every exact model exposed by the installed `tiktoken` library.
- Includes versioned model prefixes supported by `tiktoken.encoding_for_model`.
- Supports direct encoding selection from `tiktoken.list_encoding_names()`.
- Counts Gemini text locally through `google-genai` `LocalTokenizer` where the selected model is supported by the SDK.
- Estimates Claude counts without an API key using `cl100k_base`, `cl100k_base * 1.3`, and character heuristics.
- Estimates PDF files from extracted text plus provider-specific page-image tokens.
- Estimates image files from image dimensions using provider-specific visual token rules.
- Optional OCR mode can add OCR text from image files or rendered PDF pages before counting text tokens.
- Counts prompt text, each uploaded file, file totals, and combined context.
- Uses Python standard-library HTTP serving plus tokenizer libraries.

Claude estimates are approximate. Exact Claude token counting requires Anthropic API access.
Gemini local token counting is text-only, so PDF and image media counts are estimated from documented multimodal token rules.
OCR is also an estimate: it can help with scanned PDFs and screenshots, but using it on digital PDFs may double-count text that already exists as embedded PDF text.
