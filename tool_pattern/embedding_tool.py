import json
from typing import List
from tool_pattern.tool import tool  

from processors.text_processor import TextProcessor
from processors.image_processor import ImageTextProcessor

text_processor = TextProcessor()
image_processor = ImageTextProcessor() #using Teressat OCR Library
@tool
def embed_text(text: str) -> str:
    """
    Embed a given text string and return the vector.
    """
    result = text_processor.process_text(text)
    return json.dumps({
        "content": result["content"],
        "embedding": result["embedding"]
    })

@tool
def embed_image(image_path: str) -> str:
    """
    Extract text from the image and return its embedding.
    """
    extracted = image_processor.extract_text_from_images([image_path])
    if not extracted:
        return json.dumps({"error": "No text found in image."})
    return json.dumps({
        "content": extracted[0]["content"],
        "embedding": extracted[0]["embedding"]
    })

@tool
def embed_chunks(chunks: List[str]) -> str:
    """
    For each chunk (text or image path), call the appropriate embedding tool.
    """
    results = []

    for chunk in chunks:
        if chunk.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            response = embed_image(chunk)
            data = json.loads(response)
            results.append({
                "chunk": chunk,
                "type": "image",
                "content": data.get("content"),
                "embedding": data.get("embedding"),
                "error": data.get("error", None)
            })
        else:
            response = embed_text(chunk)
            data = json.loads(response)
            results.append({
                "chunk": chunk,
                "type": "text",
                "content": data.get("content"),
                "embedding": data.get("embedding"),
                "error": data.get("error", None)
            })

    return json.dumps(results)
