# AI-Powered Multimodal Storyteller

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the Application Locally](#running-the-application-locally)
- [Docker Setup](#docker-setup)
- [API Reference](#api-reference)
- [Frontend Usage](#frontend-usage)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
AI-Powered Multimodal Storyteller is a web application that generates rich, interactive stories from user‑provided prompts, images, or audio snippets. Leveraging state‑of‑the‑art transformer models from HuggingFace, the backend produces coherent narrative text, suggests visual assets, and synthesizes background music. The FastAPI backend serves a RESTful API, while the Vue.js frontend offers an intuitive UI for prompt entry, media upload, and story playback.

## Features
- **Multimodal Input**: Accept text, images, and short audio clips.
- **Story Generation**: Produce narrative paragraphs, dialogues, and scene descriptions.
- **Asset Suggestion**: Retrieve relevant images and music from public datasets via model embeddings.
- **Audio Synthesis**: Generate voice‑over narration using text‑to‑speech models.
- **Docker‑Ready**: Deploy the entire stack with a single `docker compose` command.
- **Extensible**: Plug‑in additional models or custom pipelines with minimal code changes.

## Architecture
```
┌─────────────────────┐
│   Vue.js Frontend   │   <-- HTTP (REST) & WebSocket
└─────────▲───────────┘
          │
   ┌──────┴───────┐
   │ FastAPI API  │
   │   (Python)   │
   └──────▲───────┘
          │
   ┌──────┴───────┐
   │ Transformers │   (HuggingFace models)
   │   (PyTorch)  │
   └──────────────┘
```

## Tech Stack
| Layer          | Technology                |
|----------------|---------------------------|
| Backend API    | Python 3.11, FastAPI     |
| Model Serving  | HuggingFace Transformers  |
| Frontend UI    | Vue 3 (Composition API)  |
| Containerization| Docker, Docker‑Compose   |
| Testing        | pytest, httpx, Vue Test Utils |
| CI/CD (optional) | GitHub Actions          |

## Prerequisites
- **Docker** ≥ 20.10
- **Docker Compose** ≥ 2.0
- **Python** ≥ 3.11 (required only for local development without Docker)
- **Node.js** ≥ 18 (for frontend development)

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-org/ai-powered-multimodal-storyteller.git
cd ai-powered-multimodal-storyteller
```

### 2. Backend (Python) dependencies
```bash
# Using a virtual environment is recommended
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r backend/requirements.txt
```

### 3. Frontend (Vue) dependencies
```bash
cd frontend
npm ci   # installs exact versions from package-lock.json
```

## Running the Application Locally

### Backend
```bash
# From the repository root
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
The API will be reachable at `http://localhost:8000`.

### Frontend
```bash
cd frontend
npm run dev
```
The UI will be available at `http://localhost:5173`.

### Verify the API
```bash
curl -X GET http://localhost:8000/health
# Expected response: {"status":"ok"}
```

## Docker Setup

### Build and start all services
```bash
docker compose up --build -d
```

### Stop and clean up
```bash
docker compose down --remove-orphans
```

### Service URLs (default)
- **API**: `http://localhost:8000`
- **Frontend**: `http://localhost:5173`

## API Reference

### Health Check
```
GET /health
Response: {"status":"ok"}
```

### Generate Story
```
POST /story/generate
Content-Type: multipart/form-data
Body:
  - prompt (str, required)
  - image (file, optional)
  - audio (file, optional)
Response (application/json):
{
  "story_id": "uuid",
  "text": "Generated narrative...",
  "image_urls": ["https://..."],
  "audio_url": "https://..."
}
```

### Retrieve Story
```
GET /story/{story_id}
Response: Same payload as generate endpoint
```

Full OpenAPI documentation is available at `http://localhost:8000/docs`.

## Frontend Usage

1. **Enter a prompt** – type a short description of the desired story.
2. **Upload media** – optionally attach an image or a 5‑second audio clip.
3. **Click “Generate”** – the UI calls `/story/generate` and streams progress.
4. **Read & Listen** – the generated text appears with suggested images and an audio player for narration.

The Vue components are located under `frontend/src/components/`. The main view is `StoryGenerator.vue`.

## Testing

### Backend
```bash
cd backend
pytest -v
```

### Frontend
```bash
cd frontend
npm run test:unit
```

Both test suites run in CI on every push.

## Contributing
1. Fork the repository.
2. Create a feature branch (`git checkout -b feat/awesome-feature`).
3. Write code adhering to existing style (PEP 8, ESLint).
4. Add or update tests.
5. Submit a Pull Request with a clear description and screenshots if UI changes.

Please see `CONTRIBUTING.md` for detailed guidelines.

## License
This project is licensed under the **MIT License** – see the `LICENSE` file for details.