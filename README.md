# Retail Pulse

Retail Pulse is a production-ready retail analytics app built around your
`retail_sales_dataset.csv` data. It includes:

- a training pipeline for a product category classifier
- a serialized model artifact (`retail_pulse_model.joblib`)
- a Flask web app with a polished HTML/CSS dashboard
- a JSON API endpoint for programmatic predictions

## Setup

1. Install the Python dependencies:

```bash
python -m pip install -r requirements.txt
```

2. Train the model:

```bash
python train_model.py
```

3. Start the app:

```bash
python app.py
```

4. Open your browser at `http://127.0.0.1:5000`

## Deployment

For production-ready deployment, use `gunicorn`:

```bash
gunicorn app:app --workers 2 --bind 0.0.0.0:$PORT
```

This project includes `Procfile` and `runtime.txt` for Heroku-style deployment.

## API

POST JSON to `/api/predict`:

```json
{
  "gender": "Female",
  "age": 29,
  "quantity": 2,
  "price_per_unit": 120,
  "date": "2023-08-15"
}
```

Response:

```json
{
  "product_category": "Clothing"
}
```

## Notes

- The model predicts `Product Category` from transaction features.
- The app separates training, serialization, and serving logic for maintainability.
- The UI supports browser form input and a JSON prediction endpoint.

## Docker

Build and run the container locally:

```bash
docker build -t retail-pulse:latest .
docker run -p 5000:5000 retail-pulse:latest
```

## Continuous Integration

A basic GitHub Actions workflow is included at `.github/workflows/ci.yml` which runs `pytest` on push and pull requests to `main`.
