"""FastAPI server to proxy requests to Google's Gemini API.

Simplified version — no authentication, no subscriptions, no payments.
Just a lightweight proxy that forwards requests to Gemini.
"""

import os
import logging
import httpx
import asyncio
from fastapi import FastAPI, Request, HTTPException, status, Response
from fastapi.responses import PlainTextResponse


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gemini-server")


env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(env_path):
    logger.info(f"Loading environment variables from {env_path}")
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path)
    except Exception as e:
        logger.warning(f"Could not load dotenv with python-dotenv: {e}")


app = FastAPI(
    title="Gemini Proxy Server",
    description="A lightweight FastAPI server to proxy requests to Google's Gemini API",
    version="1.0.0"
)


def parse_history_to_contents(history: list) -> list:
    """
    Parses a user-friendly conversation history into Gemini REST API's 'contents' format.
    Supports text and base64-encoded image structures.
    """
    contents = []
    for index, msg in enumerate(history):
        role = msg.get("role", "user")
        
        if role in ("assistant", "model"):
            role = "model"
        else:
            role = "user"
        
        parts = []
        
        
        text_val = msg.get("text") or msg.get("content")
        
        
        if isinstance(text_val, list):
            for part in text_val:
                if isinstance(part, dict):
                    part_type = part.get("type")
                    if part_type == "text":
                        parts.append({"text": part.get("text", "")})
                    elif part_type == "image_url":
                        img_url = part.get("image_url", {}).get("url", "")
                        if img_url.startswith("data:"):
                            try:
                                header, encoded = img_url.split(",", 1)
                                mime_type = header.split(";")[0].split(":")[1]
                                parts.append({
                                    "inlineData": {
                                        "mimeType": mime_type,
                                        "data": encoded
                                    }
                                })
                            except Exception as e:
                                logger.warning(f"Error parsing image_url data URI: {e}")
                else:
                    parts.append({"text": str(part)})
        
        elif isinstance(text_val, str) and text_val:
            parts.append({"text": text_val})
            
        
        image_val = msg.get("image")
        if image_val and isinstance(image_val, str):
            if image_val.startswith("data:"):
                try:
                    header, encoded = image_val.split(",", 1)
                    mime_type = header.split(";")[0].split(":")[1]
                    parts.append({
                        "inlineData": {
                            "mimeType": mime_type,
                            "data": encoded
                        }
                    })
                except Exception as e:
                    logger.warning(f"Error parsing image field data URI: {e}")
            else:
                parts.append({
                    "inlineData": {
                        "mimeType": "image/png",  
                        "data": image_val
                    }
                })
                
        
        images_val = msg.get("images")
        if isinstance(images_val, list):
            for img in images_val:
                if isinstance(img, str):
                    if img.startswith("data:"):
                        try:
                            header, encoded = img.split(",", 1)
                            mime_type = header.split(";")[0].split(":")[1]
                            parts.append({
                                "inlineData": {
                                    "mimeType": mime_type,
                                    "data": encoded
                                }
                            })
                        except Exception as e:
                            logger.warning(f"Error parsing images list data URI: {e}")
                    else:
                        parts.append({
                            "inlineData": {
                                "mimeType": "image/png",
                                "data": img
                            }
                        })
                        
        
        parts_val = msg.get("parts")
        if isinstance(parts_val, list):
            for p in parts_val:
                if isinstance(p, dict):
                    if "text" in p:
                        parts.append({"text": p["text"]})
                    elif "inlineData" in p:
                        parts.append({"inlineData": p["inlineData"]})
                    elif "inline_data" in p:
                        parts.append({"inlineData": p["inline_data"]})
                        
        if parts:
            contents.append({"role": role, "parts": parts})
            
    return contents

@app.post("/generate", response_class=PlainTextResponse)
async def generate(request: Request, response: Response):
    """
    POST endpoint to generate content using the Gemini API.
    Accepts JSON containing:
      - system_prompt (optional): System instruction for the model.
      - history (optional): Conversation history containing text and/or images.
      - model (optional): Specific Gemini model to use (defaults to gemini-2.5-flash).
      - contents / systemInstruction (optional): Native Gemini API payload to forward as-is.
    """
    try:
        body = await request.json()
    except Exception as e:
        logger.error(f"Failed to parse JSON body: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid JSON payload: {str(e)}"
        )

    
    input_model = body.get("model") or os.getenv("GEMINI_MODEL") or "gemini-2.5-flash"
    if input_model in ("fast", "flash", "gemini-2.5-flash"):
        model = "gemini-2.5-flash"
    elif input_model in ("thinking", "gemini-2.5-pro"):
        model = "gemini-2.5-pro"
    elif input_model in ("pro", "gemini-3.1-pro-preview"):
        model = "gemini-3.1-pro-preview"
    else:
        model = "gemini-2.5-flash"

    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("Missing GEMINI_API_KEY environment variable")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="GEMINI_API_KEY is not set in the server's environment or .env file."
        )
    
    
    gemini_payload = {}
    
    
    if "contents" in body:
        gemini_payload["contents"] = body["contents"]
        if "systemInstruction" in body:
            gemini_payload["systemInstruction"] = body["systemInstruction"]
        elif "system_prompt" in body:
            gemini_payload["systemInstruction"] = {
                "parts": [{"text": body["system_prompt"]}]
            }
    else:
        
        history = body.get("history")
        if history is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Payload must contain either 'contents' (Gemini native) or 'history'."
            )
        
        gemini_payload["contents"] = parse_history_to_contents(history)
        
        system_prompt = body.get("system_prompt")
        if system_prompt:
            gemini_payload["systemInstruction"] = {
                "parts": [{"text": system_prompt}]
            }

    
    for param in ["generationConfig", "safetySettings"]:
        if param in body:
            gemini_payload[param] = body[param]

    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    
    logger.info(f"Forwarding request to Gemini API (model: {model})")
    
    async with httpx.AsyncClient() as client:
        try:
            gemini_resp = await client.post(
                url,
                json=gemini_payload,
                headers={"Content-Type": "application/json"},
                timeout=120.0
            )
        except httpx.RequestError as exc:
            logger.error(f"HTTP request to Gemini API failed: {exc}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"An error occurred while connecting to the Gemini API: {str(exc)}"
            )

        if gemini_resp.status_code != 200:
            logger.error(f"Gemini API returned error code {gemini_resp.status_code}: {gemini_resp.text}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Gemini API returned an error ({gemini_resp.status_code}): {gemini_resp.text}"
            )

        try:
            response_json = gemini_resp.json()
            
            candidates = response_json.get("candidates", [])
            if not candidates:
                raise ValueError("No completion candidates returned from Gemini.")

            candidate = candidates[0]
            content = candidate.get("content", {})
            parts = content.get("parts", [])
            if not parts:
                raise ValueError("No content parts found in the model response.")

            
            text_response = "".join(part.get("text", "") for part in parts)
            return text_response
            
        except Exception as e:
            logger.error(f"Failed to parse Gemini response: {e}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Failed to parse Gemini API response: {str(e)}"
            )


@app.get("/health")
async def health():
    """Simple health check endpoint."""
    api_key_set = bool(os.getenv("GEMINI_API_KEY"))
    return {
        "status": "healthy",
        "gemini_api_key_configured": api_key_set
    }

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Starting server on 0.0.0.0:{port}")
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=True)
