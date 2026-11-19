"""Gemini 2.5 Flash answering with evidence-region grounding (box_2d)."""

import json
import os
import time

import numpy as np
from PIL import Image, ImageDraw


# --- Gemini client ---

def _get_client():
    """Lazy-load Gemini client."""
    from google import genai
    return genai.Client(api_key=os.environ["GOOGLE_API_KEY"])


# --- Vision RAG answering ---

VISION_PROMPT = """You are a document QA system. Answer the question using ONLY the provided document pages.

Return JSON with this exact schema:
{
  "answer": "your answer text",
  "confidence": 0.0-1.0,
  "evidence_regions": [
    {
      "page_index": 0,
      "box_2d": [ymin, xmin, ymax, xmax],
      "label": "brief description of what this region shows"
    }
  ]
}

Rules:
- box_2d coordinates are normalized to 0-1000 range
- Only cite regions that directly support your answer
- page_index is 0-based (0 = first page shown)
- If you cannot answer from these pages, set confidence to 0 and answer "Cannot determine from provided pages"

Question: {query}"""


def answer_with_evidence(query: str, page_images: list[Image.Image],
                         model_name: str = "gemini-2.5-flash") -> dict:
    """Send query + page images to Gemini, get answer with box_2d evidence.

    Returns:
        {answer, confidence, evidence_regions: [{page_index, box_2d, label}],
         latency_ms, raw_response}
    """
    client = _get_client()
    prompt = VISION_PROMPT.format(query=query)

    # Build content: prompt + images
    contents = [prompt] + page_images

    t0 = time.perf_counter()
    response = client.models.generate_content(
        model=model_name,
        contents=contents,
        config={
            "response_mime_type": "application/json",
        },
    )
    latency_ms = (time.perf_counter() - t0) * 1000

    # Parse JSON response
    raw = response.text
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Try to extract JSON from markdown code blocks
        import re
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if match:
            parsed = json.loads(match.group(1))
        else:
            parsed = {"answer": raw, "confidence": 0, "evidence_regions": []}

    return {
        "answer": parsed.get("answer", ""),
        "confidence": parsed.get("confidence", 0),
        "evidence_regions": parsed.get("evidence_regions", []),
        "latency_ms": latency_ms,
        "raw_response": raw,
    }


# --- OCR RAG answering ---

OCR_PROMPT = """You are a document QA system. Answer the question using ONLY the provided text excerpts.

Return JSON:
{{
  "answer": "your answer text",
  "confidence": 0.0-1.0,
  "cited_chunks": [0, 2]
}}

cited_chunks contains the 0-based indices of text excerpts that support your answer.
If you cannot answer, set confidence to 0.

Text excerpts:
{chunks}

Question: {query}"""


def answer_text_only(query: str, text_chunks: list[str],
                     model_name: str = "gemini-2.5-flash") -> dict:
    """Send query + text chunks to Gemini (no images).

    Returns:
        {answer, confidence, cited_chunks: [int], latency_ms, raw_response}
    """
    client = _get_client()

    chunks_formatted = "\n\n".join(
        f"[Excerpt {i}]:\n{chunk}" for i, chunk in enumerate(text_chunks)
    )
    prompt = OCR_PROMPT.format(chunks=chunks_formatted, query=query)

    t0 = time.perf_counter()
    response = client.models.generate_content(
        model=model_name,
        contents=[prompt],
        config={
            "response_mime_type": "application/json",
        },
    )
    latency_ms = (time.perf_counter() - t0) * 1000

    raw = response.text
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        import re
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if match:
            parsed = json.loads(match.group(1))
        else:
            parsed = {"answer": raw, "confidence": 0, "cited_chunks": []}

    return {
        "answer": parsed.get("answer", ""),
        "confidence": parsed.get("confidence", 0),
        "cited_chunks": parsed.get("cited_chunks", []),
        "latency_ms": latency_ms,
        "raw_response": raw,
    }


# --- Evidence overlay ---

def draw_evidence_overlay(image: Image.Image, boxes: list[dict],
                          color: str = "#FF6B35",
                          opacity: int = 80) -> Image.Image:
    """Draw semi-transparent rectangles on page image for evidence regions.

    Args:
        image: source page image
        boxes: list of {box_2d: [ymin, xmin, ymax, xmax], label: str}
               box_2d coordinates normalized to 0-1000
        color: box color as hex
        opacity: fill opacity (0-255)

    Returns:
        new image with overlays drawn
    """
    result = image.copy().convert("RGBA")
    overlay = Image.new("RGBA", result.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    w, h = image.size

    for box in boxes:
        coords = box.get("box_2d", [])
        if len(coords) != 4:
            continue

        ymin, xmin, ymax, xmax = coords

        # Normalize from 0-1000 to pixel coordinates
        x1 = int(xmin / 1000 * w)
        y1 = int(ymin / 1000 * h)
        x2 = int(xmax / 1000 * w)
        y2 = int(ymax / 1000 * h)

        # Parse hex color
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)

        # Semi-transparent fill
        draw.rectangle([x1, y1, x2, y2], fill=(r, g, b, opacity))
        # Solid border
        draw.rectangle([x1, y1, x2, y2], outline=(r, g, b, 255), width=2)

        # Label
        label = box.get("label", "")
        if label:
            draw.text((x1 + 4, y1 + 4), label[:50], fill=(r, g, b, 255))

    result = Image.alpha_composite(result, overlay)
    return result.convert("RGB")


# --- Self-check ---

if __name__ == "__main__":
    # Test overlay drawing (no API call needed)
    img = Image.new("RGB", (800, 600), "white")
    boxes = [
        {"box_2d": [100, 100, 300, 500], "label": "Revenue table"},
        {"box_2d": [400, 200, 600, 800], "label": "Chart"},
    ]
    result = draw_evidence_overlay(img, boxes)
    assert result.size == (800, 600)
    # Check that pixels were actually drawn (not all white)
    arr = np.array(result)
    assert not (arr == 255).all(), "Expected some non-white pixels"
    print("OK Evidence overlay renders correctly")

    # Test API only if key is set
    if os.environ.get("GOOGLE_API_KEY"):
        print("\nTesting Gemini API...")
        test_img = Image.new("RGB", (400, 300), "white")
        draw = ImageDraw.Draw(test_img)
        draw.text((50, 50), "Revenue: $1.2 billion", fill="black")

        result = answer_with_evidence("What is the revenue?", [test_img])
        print(f"  Answer: {result['answer']}")
        print(f"  Confidence: {result['confidence']}")
        print(f"  Evidence regions: {len(result['evidence_regions'])}")
        print(f"  Latency: {result['latency_ms']:.0f}ms")

        result_ocr = answer_text_only("What is the revenue?",
                                      ["Revenue: $1.2 billion"])
        print(f"\n  OCR answer: {result_ocr['answer']}")
        print(f"  Latency: {result_ocr['latency_ms']:.0f}ms")
    else:
        print("GOOGLE_API_KEY not set — skipping API test")

    print("\nOK answer.py self-check passed")
