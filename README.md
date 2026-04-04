# AYUDIET LLM Service and Chatbot Backend

AYUDIET is an Ayurveda-focused AI backend project that includes:
- A FastAPI microservice (`main.py`) for strict JSON integration with Node.js systems.
- A multi-agent LangGraph chatbot pipeline (`main_enhanced.py`, `agents_enhanced.py`).
- Deterministic-safe LLM utilities for schema validation, fallbacks, confidence calibration, and hallucination control.

## What is in this repo

### 1) FastAPI service for integration
`main.py` exposes production-oriented endpoints:
- `GET /health`
- `POST /ai/profile`
- `POST /ai/explain`
- `POST /ai/feedback`

Key behavior:
- strict output envelope (`success`, `data`, `error`)
- safe JSON parsing from LLM output
- timeout handling and fallback responses
- request/output/error logging

### 2) Enhanced chatbot runtime
The original chatbot stack is preserved:
- `main_enhanced.py`
- `graph_builder_enhanced.py`
- `agents_enhanced.py`
- `app.py` (Flask API)

### 3) Strict LLM control modules
- `llm_schemas_strict.py`
- `llm_prompts_strict.py`
- `llm_hallucination_control.py`
- `llm_confidence_calibration.py`
- `llm_strict_wrapper.py`

## Tech stack
- Python 3.13+
- FastAPI + Uvicorn
- Flask (legacy API path)
- LangChain / LangGraph
- Groq LLM
- FAISS + HuggingFace embeddings

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure environment
Create `.env` in project root:
```env
GROQ_API_KEY=your_groq_key
HF_TOKEN=your_hf_token
GROQ_MODEL=llama-3.1-8b-instant
LLM_TIMEOUT_SECONDS=12
```

## Run

### FastAPI (recommended for Node.js integration)
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Production entrypoint: `main.py` (FastAPI). Other runners are retained for legacy/dev use and marked deprecated.

### Flask API (legacy)
```bash
python app.py
```

### Interactive enhanced chatbot
```bash
python main_enhanced.py
```

## FastAPI request examples

### Health
```bash
curl http://localhost:8000/health
```

### Profile
```bash
curl -X POST http://localhost:8000/ai/profile \
  -H "Content-Type: application/json" \
  -d '{"text":"I have acidity and burning sensation"}'
```

### Explain
```bash
curl -X POST http://localhost:8000/ai/explain \
  -H "Content-Type: application/json" \
  -d '{"context":"Pitta is associated with heat.","reasoning":"user has heat symptoms"}'
```

### Feedback
```bash
curl -X POST http://localhost:8000/ai/feedback \
  -H "Content-Type: application/json" \
  -d '{"text":"replace this recommendation"}'
```

## Notes
- If LLM output is invalid or timeout happens, safe fallback JSON is returned.
- Keep `.env` private and do not commit secrets.
- Data files under `data/` can be large; keep repository size under control.

## License
Add your preferred license before public release.
