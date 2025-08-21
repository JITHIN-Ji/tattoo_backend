import os
from dotenv import load_dotenv
from fastapi import FastAPI, Form, File, UploadFile, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import google.generativeai as genai

from PIL import Image
from io import BytesIO
import webcolors
from starlette.responses import StreamingResponse

# Load environment variables from .env file (for GOOGLE_API_KEY)
load_dotenv()

# --- Initialize FastAPI ---
app = FastAPI(title="Ink & Soul Tattoo AI Generator API")

# --- CORS Configuration (Allows frontend to talk to backend) ---
app.add_middleware(
    CORSMiddleware,
    # Make sure this origin matches your frontend's Vercel URL exactly
    allow_origins=["https://tatoo-frontend-142t.vercel.app"],
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

# --- Reusable Core Logic (Functions to generate prompts and images) ---

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
    """Calls the Google GenAI API to generate an image."""
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables.")

        # 1. Configure the library with your API key
        genai.configure(api_key=api_key)

        #
        model = genai.GenerativeModel('gemini-2.5-flash') # Or the correct model name

        # 3. Prepare the contents for the API call
        contents = [prompt]
        if image:
            contents.append(image)

        # 4. Generate the content using the model instance
        response = model.generate_content(contents)

        # 5. Process the response to find the image data
        # The response structure for image data might differ slightly based on the model.
        # This part assumes the image data is in the first candidate's content parts.
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if part.mime_type.startswith("image/"):
                    # The image data is typically in a 'blob' for images
                    # Assuming the library provides the data directly.
                    # If it's in part.inline_data, that works too.
                    image_data = part.inline_data.data
                    return BytesIO(image_data), None

        # If no image is found, construct an error message
        error_text = "No image data was returned by the model."
        if response.prompt_feedback and response.prompt_feedback.block_reason:
            error_text = f"Request was blocked: {response.prompt_feedback.block_reason.name}"
        elif response.candidates and response.candidates[0].finish_reason:
             error_text = f"Generation finished for reason: {response.candidates[0].finish_reason.name}"


        return None, error_text
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, str(e)


class DirectPromptRequest(BaseModel):
    prompt: str

# --- API Endpoints ---

# Endpoint 1: Connects to the "AI Assistant" tab on the frontend
@app.post("/generate-detailed")
async def handle_detailed_generation(
    theme: str = Form(...), style: str = Form(...), placement: str = Form(...), vibe: str = Form(...),
    size: str = Form(...), color: str = Form(...), additions: str = Form(None), background: str = Form(None)
):
    """Endpoint for the AI Assistant (dropdowns)."""
    color_name = hex_to_color_name(color)
    prompt = create_tattoo_prompt(style, theme, color_name, placement, vibe.lower())
    if additions: prompt += f"\n\nAdditional requirements: {additions}"
    if background and background != "Clean white background": prompt += f"\n\nBackground: {background}"

    image_data, error = generate_image_from_prompt(prompt)

    if image_data:
        image_data.seek(0) # Reset buffer position to the beginning before streaming
        return StreamingResponse(image_data, media_type="image/png")
    else:
        raise HTTPException(status_code=500, detail=f"Failed to generate image: {error}")

# Endpoint 2: Connects to the "Modify a Design" tab on the frontend
@app.post("/modify-image")
async def handle_image_modification(
    prompt: str = Form(...),
    image: UploadFile = File(...)
):
    """Endpoint for modifying an uploaded image."""
    try:
        pil_image = Image.open(BytesIO(await image.read()))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file uploaded.")

    modification_prompt = f"Modify the provided tattoo image based on the following instructions: '{prompt}'. Ensure the output is a clean tattoo design suitable for an artist."

    image_data, error = generate_image_from_prompt(modification_prompt, image=pil_image)

    if image_data:
        image_data.seek(0)
        return StreamingResponse(image_data, media_type="image/png")
    else:
        raise HTTPException(status_code=500, detail=f"Failed to modify image: {error}")

# Endpoint 3: Connects to the "Direct Prompt" tab on the frontend
@app.post("/generate-direct")
async def handle_direct_generation(data: DirectPromptRequest):
    """Endpoint for the direct text prompt."""
    enhanced_prompt = f"Professional tattoo design: {data.prompt}. High quality, clean linework, suitable for actual tattooing."

    image_data, error = generate_image_from_prompt(enhanced_prompt)

    if image_data:
        image_data.seek(0)
        return StreamingResponse(image_data, media_type="image/png")
    else:
        raise HTTPException(status_code=500, detail=f"Failed to generate image: {error}")

# Root endpoint for checking if the server is running
@app.get("/")
def read_root():
    return {"status": "Ink & Soul AI Generator is running!"}
