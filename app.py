import os
import logging
from flask import Flask, request, jsonify, render_template
from retail_pulse_pipeline import load_model, build_input_row, predict_category

MODEL_PATH = "retail_pulse_model.joblib"
app = Flask(__name__, template_folder="templates", static_folder="static")

# configure logging for production
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("retail_pulse")


def get_model():
    try:
        return load_model(MODEL_PATH)
    except FileNotFoundError:
        raise RuntimeError("No trained model found. Run train_model.py before starting the app.")


MODEL = get_model()


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    values = {
        "gender": "Male",
        "age": 30,
        "quantity": 1,
        "price_per_unit": 50.0,
        "date": "2023-01-01",
    }

    if request.method == "POST":
        values = {
            "gender": request.form.get("gender", "Male"),
            "age": int(request.form.get("age", 30)),
            "quantity": int(request.form.get("quantity", 1)),
            "price_per_unit": float(request.form.get("price_per_unit", 50)),
            "date": request.form.get("date", "2023-01-01"),
        }
        payload = build_input_row(**values)
        prediction = predict_category(MODEL, payload)

    return render_template("index.html", prediction=prediction, values=values)


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


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint used by load balancers and orchestration."""
    try:
        # simple model existence check
        _ = MODEL is not None
        return jsonify({"status": "ok"}), 200
    except Exception as e:
        logger.exception("Health check failed")
        return jsonify({"status": "error", "detail": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
