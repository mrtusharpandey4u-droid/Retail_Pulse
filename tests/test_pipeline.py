from retail_pulse_pipeline import build_input_row, load_model, predict_category


def test_pipeline_predict():
    model = load_model("retail_pulse_model.joblib")
    row = build_input_row(gender="Male", age=35, quantity=2, price_per_unit=99.99, date="2023-01-01")
    pred = predict_category(model, row)
    assert isinstance(pred, str)
