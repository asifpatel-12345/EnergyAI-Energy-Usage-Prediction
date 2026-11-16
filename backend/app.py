"""Flask backend: serves the frontend and the prediction API."""
import os
from flask import Flask, jsonify, request, send_from_directory
from model import EnergyModel, recommendations, RATE_PER_KWH, CO2_PER_KWH

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(os.path.dirname(BASE_DIR), "frontend")

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
print("Training model, please wait...")
ml = EnergyModel()
print(f"Model ready. R2={ml.r2}, MAE={ml.mae} kWh")


def num(data, key, default, lo, hi, cast=float):
    """Safe number reader with validation."""
    try:
        value = cast(data.get(key, default))
    except (TypeError, ValueError):
        raise ValueError(f"'{key}' must be a number")
    if value < lo or value > hi:
        raise ValueError(f"'{key}' must be between {lo} and {hi}")
    return value


@app.route("/")
def home():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "r2": ml.r2, "mae": ml.mae})


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    try:
        temp = num(data, "temperature", 30, -10, 55)
        occ = num(data, "occupants", 3, 1, 20, int)
        appl = num(data, "appliances", 4, 0, 20, int)
        weekend = num(data, "is_weekend", 0, 0, 1, int)
    except ValueError as err:
        return jsonify({"error": str(err)}), 400

    base = ml.forecast_day(temp, occ, appl, weekend)
    opt = ml.optimized_day(temp, occ, appl, weekend)
    total, total_opt = sum(base), sum(opt)
    saved = max(0.0, total - total_opt)

    return jsonify({
        "hours": list(range(24)),
        "forecast": base,
        "optimized": opt,
        "total_kwh": round(total, 2),
        "optimized_kwh": round(total_opt, 2),
        "saved_kwh": round(saved, 2),
        "saved_percent": round(saved / total * 100, 1) if total else 0,
        "cost": round(total * RATE_PER_KWH, 2),
        "saved_cost": round(saved * RATE_PER_KWH, 2),
        "co2_saved": round(saved * CO2_PER_KWH, 2),
        "monthly_saving": round(saved * RATE_PER_KWH * 30, 2),
        "tips": recommendations(temp, occ, appl, weekend, base, opt),
        "model": {"r2": ml.r2, "mae": ml.mae},
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
