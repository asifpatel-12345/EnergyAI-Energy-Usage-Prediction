# ⚡ EnergyAI - Energy Usage Prediction & Optimization

An AI-based web app that predicts 24-hour electricity usage of a home/business and gives energy-saving recommendations.
Built for the college Hackathon (AIML problem statement #6).

## Features
- Hourly + daily energy prediction (kWh)
- Cost, monthly saving and CO2 saving estimate
- Predicted vs optimized usage graph (offline canvas chart)
- Smart recommendations (peak hours, AC setting, appliance shifting)

## Tech Stack
Python, Flask, NumPy (Linear Regression model), HTML, CSS, JavaScript

## Run in VS Code
1. Open the `energy-ai` folder in VS Code
2. Terminal: `cd backend`
3. `python -m pip install -r requirements.txt`
4. `python app.py`
5. Open http://127.0.0.1:5000

## Project Structure
```
energy-ai/
├── backend/
│   ├── app.py            Flask API (/api/predict, /api/health) + serves frontend
│   ├── model.py          data generation, regression model, recommendations
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
├── .gitignore
└── README.md
```

## How it works
The model is trained on synthetic data (hour, temperature, people, appliances, weekday/weekend).
Future scope: real smart-meter data, IoT sensors, bill upload, user login.
