# Together Synthetic Persona App

Robust, low-cost Streamlit app using Together API.

## Why this version

- Backend-managed key only (`TOGETHER_API_KEY`) — no user key input.
- Cheap default models preconfigured.
- Sequential calls + retries to reduce rate-limit failures.
- Graceful partial-results behavior when some persona calls fail.

## Local run

```bash
pip install -r requirements.txt
```

Create `.streamlit/secrets.toml`:

```toml
TOGETHER_API_KEY = "your-key"
```

Run:

```bash
streamlit run app.py
```

## Streamlit Cloud

- Branch: `main`
- Main file: `app.py`
- Python: `3.11`
- Secrets:

```toml
TOGETHER_API_KEY = "your-key"
```
