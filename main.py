import os
from dotenv import load_dotenv
from fastapi import FastAPI, Form, File, UploadFile, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import google.generativeai as genai
from google.generativeai import types
from PIL import Image
from io import BytesIO
import webcolors
from starlette.responses import StreamingResponse

# Load environment variables from .env file (for GOOGLE_API_KEY)
load_dotenv()

# --- Configure Gemini ---
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")
genai.configure(api_key=api_key)

# --- Initialize FastAPI ---
app = FastAPI(title="Ink & Soul Tattoo AI Generator API")

# --- CORS Configuration (Allows frontend to talk to backend) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, restrict this to your actual frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Custom Exception Handler (for clean validation errors) ---
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    error = exc.errors()[0]
    field = " -> ".join(map(str, error['loc']))
    message = f"Validation Error: Field '{field}' - {error['msg']}"
    return JSONResponse(status_code=422, content={"detail": message})

# --- Utility Functions ---
def hex_to_color_name(hex_code):
    """Converts a HEX color code to its nearest color name."""
    try:
        return webcolors.hex_to_name(hex_code)
    except ValueError:
        return "custom color"

def create_tattoo_prompt(style, theme, color_name, placement, vibe):
    """Creates the detailed prompt for the AI model from structured data."""
    return f"""Create a professional {style.lower()} tattoo design of a {theme} in {color_name} ink.
    This tattoo is designed for placement on the {placement.lower()} with a {vibe} aesthetic.
    Style specifications: Clean, professional linework suitable for actual tattooing, {style} artistic style, high contrast and clear details.
    Please generate a high-quality tattoo design that a professional tattoo artist could use as reference."""

def generate_image_from_prompt(prompt: str, image: Image.Image = None):
    """Calls the Google Generative AI API to generate an image."""
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")  # text + image model
        contents = [prompt]
        if image:
            contents.append(image)

        response = model.generate_content(
            contents,
            generation_config=types.GenerationConfig(
                response_mime_type="image/png"
            )
        )

        # Extract image bytes
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                return BytesIO(part.inline_data.data), None

        error_text = response.candidates[0].content.parts[0].text if response.candidates else "No image data returned."
        return None, error_text

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, str(e)

# --- Request Models ---
class DirectPromptRequest(BaseModel):
    prompt: str

# --- API Endpoints ---
@app.post("/generate-detailed")
async def handle_detailed_generation(
    theme: str = Form(...), style: str = Form(...), placement: str = Form(...), vibe: str = Form(...),
    size: str = Form(...), color: str = Form(...), additions: str = Form(None), background: str = Form(None)
):
    """Endpoint for the AI Assistant (dropdowns)."""
    color_name = hex_to_color_name(color)
    prompt = create_tattoo_prompt(style, theme, color_name, placement, vibe.lower())
    if additions:
        prompt += f"\n\nAdditional requirements: {additions}"
    if background and background != "Clean white background":
        prompt += f"\n\nBackground: {background}"

    image_data, error = generate_image_from_prompt(prompt)
    if image_data:
        return StreamingResponse(image_data, media_type="image/png")
    else:
        raise HTTPException(status_code=500, detail=f"Failed to generate image: {error}")

@app.post("/modify-image")
async def handle_image_modification(
    prompt: str = Form(...),
    image: UploadFile = File(...),
):
    """Endpoint for modifying an uploaded image."""
    pil_image = Image.open(BytesIO(await image.read()))
    modification_prompt = f"Modify the provided tattoo image based on the following instructions: '{prompt}'. Ensure the output is a clean tattoo design suitable for an artist."

    image_data, error = generate_image_from_prompt(modification_prompt, image=pil_image)
    if image_data:
        return StreamingResponse(image_data, media_type="image/png")
    else:
        raise HTTPException(status_code=500, detail=f"Failed to modify image: {error}")

@app.post("/generate-direct")
async def handle_direct_generation(data: DirectPromptRequest):
    """Endpoint for the direct text prompt."""
    enhanced_prompt = f"Professional tattoo design: {data.prompt}. High quality, clean linework, suitable for actual tattooing."

    image_data, error = generate_image_from_prompt(enhanced_prompt)
    if image_data:
        return StreamingResponse(image_data, media_type="image/png")
    else:
        raise HTTPException(status_code=500, detail=f"Failed to generate image: {error}")

@app.get("/")
def read_root():
    return {"status": "Ink & Soul AI Generator is running!"}
