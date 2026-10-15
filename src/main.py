import os
import logging
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, PipelineException

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
MODEL_NAME = os.getenv("STORYTELLER_MODEL", "gpt2")  # default to GPT-2
MAX_NEW_TOKENS = int(os.getenv("STORYTELLER_MAX_TOKENS", "150"))
DEVICE = int(os.getenv("STORYTELLER_DEVICE", "-1"))  # -1 = CPU, >=0 = CUDA device id

# ----------------------------------------------------------------------
# Logging setup
# ----------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("storyteller")

# ----------------------------------------------------------------------
# Pydantic models
# ----------------------------------------------------------------------
class StoryRequest(BaseModel):
    """Request payload for story generation."""

    prompt: str = Field(..., description="The initial text prompt for story generation.")
    max_new_tokens: Optional[int] = Field(
        None,
        ge=1,
        le=1024,
        description="Maximum number of tokens to generate. Defaults to server config.",
    )

    @validator("prompt")
    def non_empty_prompt(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Prompt must not be empty.")
        return v


class StoryResponse(BaseModel):
    """Response payload containing the generated story."""

    story: str = Field(..., description="The generated story text.")
    prompt: str = Field(..., description="The original prompt used for generation.")
    max_new_tokens: int = Field(..., description="Number of tokens generated.")


class ErrorResponse(BaseModel):
    """Standard error response."""

    detail: str = Field(..., description="Human‑readable error description.")


# ----------------------------------------------------------------------
# FastAPI app definition
# ----------------------------------------------------------------------
app = FastAPI(
    title="AI‑Powered Multimodal Storyteller",
    description="Generate creative stories from textual prompts using HuggingFace Transformers.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow all origins for simplicity; adjust for production as needed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------------------------
# Global model objects (initialized on startup)
# ----------------------------------------------------------------------
tokenizer = None
text_generator = None


@app.on_event("startup")
async def load_model() -> None:
    """Load the tokenizer and model pipeline at application start."""
    global tokenizer, text_generator
    try:
        logger.info("Loading tokenizer for model %s", MODEL_NAME)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        logger.info("Loading causal language model %s on device %s", MODEL_NAME, DEVICE)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

        # Move model to the appropriate device if CUDA is requested
        if DEVICE >= 0:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA device requested but not available.")
            model.to(f"cuda:{DEVICE}")

        logger.info("Creating text generation pipeline")
        text_generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=DEVICE,
        )
        logger.info("Model and tokenizer loaded successfully.")
    except Exception as exc:
        logger.exception("Failed to load model: %s", exc)
        raise RuntimeError(f"Model loading failed: {exc}") from exc


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Clean up resources on shutdown."""
    global tokenizer, text_generator
    logger.info("Shutting down application, releasing model resources.")
    tokenizer = None
    text_generator = None


# ----------------------------------------------------------------------
# Exception handlers
# ----------------------------------------------------------------------
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    logger.warning("HTTPException %s: %s", exc.status_code, exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(detail=exc.detail).dict(),
    )


@app.exception_handler(PipelineException)
async def pipeline_exception_handler(request: Request, exc: PipelineException) -> JSONResponse:
    logger.error("PipelineException: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(detail="Internal generation error.").dict(),
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(detail="Unexpected server error.").dict(),
    )


# ----------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------
@app.get("/health", summary="Health check endpoint")
async def health_check() -> dict:
    """Simple health check used by orchestration tools."""
    return {"status": "ok"}


@app.post(
    "/generate",
    response_model=StoryResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Generate a story from a text prompt",
)
async def generate_story(payload: StoryRequest) -> StoryResponse:
    """
    Generate a creative story using the configured language model.

    Parameters
    ----------
    payload: StoryRequest
        The request containing the prompt and optional generation length.

    Returns
    -------
    StoryResponse
        The generated story together with metadata.
    """
    if text_generator is None:
        logger.error("Generation requested before model was loaded.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded yet. Try again later.",
        )

    max_tokens = payload.max_new_tokens or MAX_NEW_TOKENS
    logger.info("Generating story for prompt (max %d tokens)", max_tokens)

    try:
        generated = text_generator(
            payload.prompt,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            num_return_sequences=1,
        )
        story_text = generated[0]["generated_text"]
    except Exception as exc:
        logger.exception("Generation failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Story generation failed.",
        ) from exc

    # Trim the original prompt from the generated text to avoid duplication
    if story_text.startswith(payload.prompt):
        story_text = story_text[len(payload.prompt) :].lstrip()

    response = StoryResponse(
        story=story_text,
        prompt=payload.prompt,
        max_new_tokens=max_tokens,
    )
    logger.debug("Generated story: %s", response.story[:100])
    return response


# ----------------------------------------------------------------------
# Entry point for `python -m src.main` execution
# ----------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        log_level="info",
        reload=os.getenv("DEV_MODE", "false").lower() == "true",
    )