import os
import base64

from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from openai import OpenAI


# Load environment variables
load_dotenv()


# Flask application
app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static"
)

# Maximum upload size: 25 MB
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024


# ============================================================
# AZURE OPENAI SETTINGS
# ============================================================

AOAI_ENDPOINT = os.getenv(
    "AZURE_OPENAI_ENDPOINT",
    ""
).rstrip("/")

AOAI_KEY = os.getenv(
    "AZURE_OPENAI_API_KEY",
    ""
)

TEXT_MODEL = os.getenv(
    "TEXT_MODEL_DEPLOYMENT",
    ""
)

VISION_MODEL = os.getenv(
    "VISION_MODEL_DEPLOYMENT",
    ""
) or TEXT_MODEL


# ============================================================
# AZURE OPENAI CLIENT
# ============================================================

def client():

    if not AOAI_ENDPOINT or not AOAI_KEY:
        raise RuntimeError(
            "Set AZURE_OPENAI_ENDPOINT and "
            "AZURE_OPENAI_API_KEY in .env"
        )

    endpoint = AOAI_ENDPOINT.removesuffix(
        "/openai/v1"
    ).rstrip("/")

    return OpenAI(
        api_key=AOAI_KEY,
        base_url=f"{endpoint}/openai/v1/"
    )


# ============================================================
# TEXT RESPONSE
# ============================================================

def text_response(
    prompt,
    system="""You are a Travel Document Assistant.

You help users understand travel documents such as:

- Passports
- Visas
- Travel permits
- Entry requirements
- Document validity
- Travel document information

Give simple and clear answers.

Do not guess or invent information.

If information may depend on the user's country,
destination, nationality, or current government rules,
tell the user to verify it with the official government
or embassy website.
"""
):

    response = client().responses.create(
        model=TEXT_MODEL,
        instructions=system,
        input=prompt
    )

    return response.output_text


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():
    return render_template("home.html")


# ============================================================
# TRAVEL DOCUMENT CHAT
# ============================================================

@app.get("/chat")
def chat():
    return render_template("chat.html")


@app.post("/api/chat")
def api_chat():

    try:

        message = (
            request.json or {}
        ).get("message", "").strip()

        if not message:
            return jsonify(
                error="Enter a message."
            ), 400

        reply = text_response(message)

        return jsonify(
            reply=reply
        )

    except Exception as e:

        return jsonify(
            error=str(e)
        ), 500


# ============================================================
# TRAVEL DOCUMENT ANALYSIS
# ============================================================

@app.get("/document-analysis")
def document_analysis():

    return render_template(
        "document_analysis.html"
    )


@app.post("/api/document-analysis")
def api_document_analysis():

    try:

        # Get uploaded image
        f = request.files.get("image")

        if not f:
            return jsonify(
                error="Upload a document image."
            ), 400

        # Read and encode image
        data = base64.b64encode(
            f.read()
        ).decode()

        mime = f.mimetype or "image/jpeg"

        # Prompt specifically for travel documents
        prompt = """
You are a Travel Document Assistant.

Analyze the uploaded travel document image.

Try to identify the following information ONLY
when it is clearly visible:

1. Document type
2. Country
3. Full name
4. Document number
5. Date of birth
6. Issue date
7. Expiry date
8. Visa type or category
9. Important notes or restrictions

Return the result in a simple and readable format.

For example:

Document Type:
Country:
Name:
Document Number:
Date of Birth:
Issue Date:
Expiry Date:
Visa Type:
Important Notes:

If a field is not visible or cannot be read clearly,
write:

Not clearly visible

IMPORTANT:
Do not guess.
Do not create missing information.
Only report information that can actually be seen
in the image.
"""

        response = client().responses.create(
            model=VISION_MODEL,

            input=[
                {
                    "role": "user",
                    "content": [

                        {
                            "type": "input_text",
                            "text": prompt
                        },

                        {
                            "type": "input_image",
                            "image_url": (
                                f"data:{mime};base64,{data}"
                            )
                        }

                    ]
                }
            ]
        )

        return jsonify(
            result=response.output_text
        )

    except Exception as e:

        return jsonify(
            error=str(e)
        ), 500


# ============================================================
# VISION
# ============================================================

@app.get("/vision")
def vision():

    return render_template(
        "vision.html"
    )


@app.post("/api/vision")
def api_vision():

    try:

        # Get uploaded image
        f = request.files.get("image")

        # Get user's prompt
        prompt = request.form.get(
            "prompt",
            """
Analyze this travel-related image.

Describe what you can see.
Identify any travel-related objects,
documents, signs, locations, or visible text.

Do not guess information that cannot
be clearly seen.
"""
        ).strip()

        if not f:
            return jsonify(
                error="Upload an image."
            ), 400

        # Convert image to Base64
        data = base64.b64encode(
            f.read()
        ).decode()

        mime = f.mimetype or "image/jpeg"

        # Send image to Azure OpenAI Vision
        response = client().responses.create(
            model=VISION_MODEL,

            input=[
                {
                    "role": "user",
                    "content": [

                        {
                            "type": "input_text",
                            "text": prompt
                        },

                        {
                            "type": "input_image",
                            "image_url": (
                                f"data:{mime};base64,{data}"
                            )
                        }

                    ]
                }
            ]
        )

        return jsonify(
            result=response.output_text
        )

    except Exception as e:

        return jsonify(
            error=str(e)
        ), 500


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return jsonify(
        text_model=bool(TEXT_MODEL),
        vision_model=bool(VISION_MODEL)
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=False,
        host="127.0.0.1",
        port=5000
    )