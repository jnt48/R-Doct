from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import base64
import io
from PIL import Image
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

app = FastAPI(title="Document Data Extractor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Add your frontend URL here
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],
)

def get_gemini_response(prompt: str, image_content: dict) -> str:
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content([prompt, image_content])
    return response.text

def input_image_setup(file_bytes: bytes) -> dict:

    try:
        image = Image.open(io.BytesIO(file_bytes))

        if image.mode != "RGB":
            image = image.convert("RGB")
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="JPEG")
        img_bytes = img_byte_arr.getvalue()

        image_data = {
            "mime_type": "image/jpeg",
            "data": base64.b64encode(img_bytes).decode()
        }
        return image_data
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {e}")

@app.post("/extract")
async def extract_data(
    file: UploadFile = File(...),
    instructions: str = Form("")
):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    try:
        file_bytes = await file.read()
        image_content = input_image_setup(file_bytes)

        extraction_prompt = (
            "You are an expert document data extractor. Analyze the provided image of a document of any type "
            "(e.g., form, invoice, letter, etc.) and identify all relevant fields and information. "
            "Return only the extracted data in a valid JSON format with descriptive keys and no additional text or markdown formatting. "
            "Do not include any triple backticks or code fences. Include no \\n in the code"

        )
        if instructions.strip():
            extraction_prompt += "\nAdditional instructions: " + instructions.strip()

        extracted_data = get_gemini_response(extraction_prompt, image_content)
        return {"extracted_data": extracted_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)