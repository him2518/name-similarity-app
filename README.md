# Name Similarity App

Name comparison engine with Streamlit UI and FastAPI endpoint.

## Streamlit UI

```bash
pip install -r requirements.txt
streamlit run app.py
```

## API

```bash
pip install -r requirements.txt
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

### API request

`POST /match`

```json
{
  "name_1": "Mohammad Faizan Shaikh",
  "name_2": "Mohd Faizan Sheikh"
}
```

### API response

Returns JSON with:
- all algorithms with their score
- average score
- weighted score
- normalization details
- decision hint
