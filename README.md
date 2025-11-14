# Autonomous Transaction Intelligence (ATI) Engine

FastAPI microservice with DistilBERT inference, SHAP explainability, YAML-based taxonomy mapping, Streamlit demo UI, and Docker packaging. Designed to be stateless, scalable, and cleanly modular.

## Architecture

- API (FastAPI): REST endpoints for health, inference, explain, taxonomy.
- Preprocessing: Text normalization utilities.
- Inference: DistilBERT (zero-shot) wrapper and service orchestration.
- Taxonomy: YAML loader and label listing.
- XAI: SHAP token-level attribution.
- Streamlit: Simple demo UI that calls the API.

```
ati_engine/
  api/
  core/
  preprocessing/
  inference/
  taxonomy/
  xai/
streamlit_app/
```

## Quickstart (local)

1. Create a virtual environment and install dependencies.
2. Run the API, then the Streamlit UI.

```powershell
# Windows PowerShell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
uvicorn ati_engine.api.main:app --host 0.0.0.0 --port 8000 --reload
# In another terminal
streamlit run streamlit_app/app.py
```

Open http://localhost:8000/docs for API docs and http://localhost:8501 for the UI.

## Docker

Build and run both services using Docker Compose:

```powershell
docker compose up --build
```

- API: http://localhost:8000
- Streamlit: http://localhost:8501

## Configuration

Environment variables (with defaults):

- `MODEL_NAME`: DistilBERT MNLI model for zero-shot (default: `typeform/distilbert-base-uncased-mnli`).
- `DEVICE`: Inference device id (`-1` for CPU).
- `TAXONOMY_PATH`: Path to taxonomy YAML (default: `ati_engine/taxonomy/sample_taxonomy.yaml`).
- `LOG_LEVEL`: Logging level (default: `INFO`).

Create `.env` based on `.env.example` for local overrides.

## Taxonomy

A simple YAML structure mapping high-level categories and optional `subcategories`.
Update `ati_engine/taxonomy/sample_taxonomy.yaml` to suit your domain.

## Notes on Explainability

SHAP is applied with a simple wrapper around the service prediction focusing on the probability of the target label. For production, consider exposing full per-label probabilities from the model and optimizing SHAP runtime.

## Testing

```powershell
pytest -q
```

The included tests are lightweight and avoid loading large models.

## Production tips

- Use GPU by setting `DEVICE` to a CUDA device id.
- Pre-pull model artifacts in your Docker image for faster cold starts.
- Scale using multiple replicas behind a load balancer; the service is stateless.
