import os
from flask import Flask, request, jsonify, render_template_string
from retail_pulse_pipeline import load_model, build_input_row, predict_category

MODEL_PATH = "retail_pulse_model.joblib"
app = Flask(__name__)


def get_model():
    try:
        return load_model(MODEL_PATH)
    except FileNotFoundError:
        raise RuntimeError("No trained model found. Run train_model.py before starting the app.")


MODEL = get_model()

HTML_FORM = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>Retail Pulse</title>
    <style>
      :root {
        --bg-page: #f4f7fb;
        --bg-panel: #ffffff;
        --text-primary: #102a43;
        --text-secondary: #486581;
        --border: #d9e2ec;
        --accent: #1766dc;
        --accent-soft: #e7f0ff;
        --surface: #ffffff;
        --shadow: 0 24px 80px rgba(16, 42, 67, 0.08);
      }

      * {
        box-sizing: border-box;
      }

      body {
        margin: 0;
        min-height: 100vh;
        font-family: Inter, 'Segoe UI', sans-serif;
        background: radial-gradient(circle at top left, rgba(23, 102, 220, 0.12), transparent 24%),
                    radial-gradient(circle at bottom right, rgba(56, 189, 248, 0.12), transparent 20%),
                    var(--bg-page);
        color: var(--text-primary);
      }

      .container {
        width: min(100%, 1100px);
        margin: 0 auto;
        padding: 36px 24px 48px;
      }

      header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 18px;
        margin-bottom: 32px;
      }

      .brand {
        display: grid;
        gap: 8px;
      }

      .brand__name {
        margin: 0;
        font-size: clamp(1.9rem, 2.5vw, 2.8rem);
        letter-spacing: -0.04em;
      }

      .brand__tagline {
        margin: 0;
        color: var(--text-secondary);
        font-size: 1rem;
      }

      .panel {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 28px;
        box-shadow: var(--shadow);
        overflow: hidden;
      }

      .panel__top {
        padding: 32px 32px 24px;
      }

      .panel__title {
        margin: 0 0 12px;
        font-size: 1.35rem;
      }

      .panel__description {
        margin: 0;
        color: var(--text-secondary);
        line-height: 1.75;
      }

      .panel__body {
        display: grid;
        gap: 24px;
        padding: 0 32px 32px;
      }

      .grid {
        display: grid;
        gap: 24px;
      }

      .grid--columns {
        grid-template-columns: minmax(0, 1.2fr) minmax(320px, 0.8fr);
      }

      .form-card,
      .info-card {
        padding: 24px;
        border: 1px solid var(--border);
        border-radius: 22px;
        background: var(--bg-panel);
      }

      .form-card h2,
      .info-card h2 {
        margin: 0 0 18px;
        font-size: 1.15rem;
      }

      .form-grid {
        display: grid;
        gap: 18px;
      }

      label {
        display: grid;
        gap: 10px;
        font-weight: 600;
        color: var(--text-primary);
      }

      input,
      select {
        width: 100%;
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 14px 16px;
        background: #f8fbff;
        color: var(--text-primary);
        font-size: 0.95rem;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
      }

      input:focus,
      select:focus {
        border-color: rgba(23, 102, 220, 0.45);
        box-shadow: 0 0 0 6px rgba(23, 102, 220, 0.12);
        outline: none;
      }

      .button-row {
        margin-top: 14px;
      }

      button {
        width: 100%;
        max-width: 260px;
        padding: 16px 24px;
        border: none;
        border-radius: 999px;
        color: white;
        background: linear-gradient(135deg, #1766dc, #05a3ff);
        font-size: 1rem;
        font-weight: 700;
        cursor: pointer;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
      }

      button:hover {
        transform: translateY(-1px);
        box-shadow: 0 18px 40px rgba(23, 102, 220, 0.18);
      }

      .notice {
        display: inline-flex;
        align-items: center;
        gap: 12px;
        margin-top: 12px;
        padding: 14px 18px;
        border-radius: 18px;
        background: #eff7ff;
        color: #0f172a;
        border: 1px solid rgba(23, 102, 220, 0.16);
      }

      .result-card {
        padding: 24px;
        border-radius: 22px;
        background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
        border: 1px solid var(--border);
      }

      .result-title {
        margin: 0 0 10px;
        font-size: 1rem;
        color: #334e68;
      }

      .prediction {
        padding: 18px 22px;
        border-radius: 18px;
        background: linear-gradient(135deg, #e7f0ff, #d6e9ff);
        color: #0f172a;
        font-size: 1.4rem;
        font-weight: 700;
        text-align: center;
        border: 1px solid rgba(23, 102, 220, 0.18);
      }

      .stats {
        display: grid;
        gap: 16px;
        margin-top: 6px;
      }

      .stat {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 14px 16px;
        border-radius: 18px;
        background: #f8fbff;
        border: 1px solid rgba(15, 118, 232, 0.1);
      }

      .stat strong {
        color: #102a43;
        font-size: 0.97rem;
      }

      .stat span {
        color: #486581;
      }

      .footer {
        margin-top: 18px;
        color: var(--text-secondary);
        font-size: 0.95rem;
      }

      @media (max-width: 900px) {
        .grid--columns {
          grid-template-columns: 1fr;
        }
      }
    </style>
  </head>
  <body>
    <div class="container">
      <header>
        <div class="brand">
          <h1 class="brand__name">Retail Pulse</h1>
          <p class="brand__tagline">A professional retail forecasting interface for product category prediction.</p>
        </div>
        <div class="notice">Ready to predict product categories from customer and purchase details.</div>
      </header>

      <section class="panel">
        <div class="panel__top">
          <h2 class="panel__title">Predict with confidence</h2>
          <p class="panel__description">Enter transaction details below and let the model estimate the product category based on purchase attributes and customer profile.</p>
        </div>

        <div class="panel__body grid grid--columns">
          <div class="form-card">
            <h2>Transaction input</h2>
            <form method="post" action="/predict">
              <div class="form-grid">
                <label>
                  Gender
                  <select name="gender">
                    <option>Male</option>
                    <option>Female</option>
                  </select>
                </label>

                <label>
                  Age
                  <input type="number" name="age" min="1" max="120" value="30" required />
                </label>

                <label>
                  Quantity
                  <input type="number" name="quantity" min="1" max="100" value="1" required />
                </label>

                <label>
                  Price per Unit
                  <input type="number" step="0.01" name="price_per_unit" min="1" max="10000" value="50" required />
                </label>

                <label>
                  Date
                  <input type="date" name="date" value="2023-01-01" required />
                </label>
              </div>

              <div class="button-row">
                <button type="submit">Predict Category</button>
              </div>
            </form>
          </div>

          <div class="info-card">
            <h2>Model output</h2>
            {% if prediction %}
              <div class="result-card">
                <p class="result-title">Predicted Product Category</p>
                <div class="prediction">{{ prediction }}</div>
              </div>
            {% else %}
              <p class="panel__description">The prediction will display here after form submission. Use the input panel on the left to generate a result.</p>
            {% endif %}

            <div class="stats">
              <div class="stat">
                <strong>Target</strong>
                <span>Product Category</span>
              </div>
              <div class="stat">
                <strong>Data size</strong>
                <span>1000 records</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      <div class="footer">Professional viewing with a clean dashboard, responsive layout, and clear model output.</div>
    </div>
  </body>
</html>
"""


@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_FORM)


@app.route("/predict", methods=["POST"])
def predict():
    values = {
        "gender": request.form.get("gender", "Male"),
        "age": request.form.get("age", "30"),
        "quantity": request.form.get("quantity", "1"),
        "price_per_unit": request.form.get("price_per_unit", "50"),
        "date": request.form.get("date", "2023-01-01"),
    }
    payload = build_input_row(
        gender=values["gender"],
        age=int(values["age"]),
        quantity=int(values["quantity"]),
        price_per_unit=float(values["price_per_unit"]),
        date=values["date"],
    )
    prediction = predict_category(MODEL, payload)
    return render_template_string(HTML_FORM, prediction=prediction)


@app.route("/api/predict", methods=["POST"])
def predict_api():
    data = request.get_json(force=True)
    payload = build_input_row(
        gender=data.get("gender", "Male"),
        age=int(data.get("age", 30)),
        quantity=int(data.get("quantity", 1)),
        price_per_unit=float(data.get("price_per_unit", 50)),
        date=data.get("date", None),
    )
    prediction = predict_category(MODEL, payload)
    return jsonify({"product_category": prediction})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
