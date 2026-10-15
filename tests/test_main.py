"""Tests for the FastAPI application of the AI-Powered Multimodal Storyteller.

These tests cover the health endpoint and the story generation endpoint. The heavy
model inference is mocked so the tests run quickly and deterministically.
"""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

# Import the FastAPI app defined in the project.
# The project is expected to expose a FastAPI instance named ``app`` in
# ``app/main.py``.
from app.main import app

# Create a test client that will be used for all requests.
client = TestClient(app)


def test_health_check() -> None:
    """The ``/health`` endpoint must return a simple JSON status."""
    response = client.get("/health")
    assert response.status_code == 200, "Health endpoint should return HTTP 200"
    json_body = response.json()
    assert isinstance(json_body, dict), "Response body must be a JSON object"
    assert json_body.get("status") == "ok", "Health status should be 'ok'"


@patch("app.services.story_generator.generate_story")
def test_generate_story_success(mock_generate_story) -> None:
    """A valid request to ``/generate`` should return the mocked story text."""
    # Arrange: configure the mock to return a deterministic story.
    mock_story = "Once upon a time, in a faraway land, a brave hero emerged."
    mock_generate_story.return_value = mock_story

    payload = {
        "prompt": "A brave hero emerges",
        "image_url": "http://example.com/hero.png",
    }

    # Act: send the request to the API.
    response = client.post("/generate", json=payload)

    # Assert: the endpoint returns HTTP 200 and the mocked story.
    assert response.status_code == 200, "Generate endpoint should succeed with 200"
    json_body = response.json()
    assert isinstance(json_body, dict), "Response must be a JSON object"
    assert json_body.get("story") == mock_story, "Returned story must match the mock"

    # Verify that the service was called with the correct arguments.
    mock_generate_story.assert_called_once_with(
        prompt=payload["prompt"], image_url=payload["image_url"]
    )


def test_generate_story_invalid_input() -> None:
    """Missing required fields should cause a validation error (HTTP 422)."""
    # Omit the ``prompt`` field which is required by the Pydantic model.
    payload = {"image_url": "http://example.com/hero.png"}

    response = client.post("/generate", json=payload)

    assert response.status_code == 422, "Invalid payload should result in 422 Unprocessable Entity"
    json_body = response.json()
    # FastAPI returns a list of errors under the ``detail`` key.
    assert "detail" in json_body, "Error response must contain 'detail' field"
    # Ensure the error mentions the missing ``prompt`` field.
    error_messages = [err.get("msg", "") for err in json_body["detail"]]
    assert any("field required" in msg for msg in error_messages), "Error should indicate missing required fields"
