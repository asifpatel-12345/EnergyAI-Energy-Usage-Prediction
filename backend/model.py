"""ML part: synthetic data + RandomForest energy predictor + recommendations."""
import numpy as np

FEATURES = ["hour", "temperature", "occupants", "appliances", "is_weekend"]
RATE_PER_KWH = 8.0      # Rs per kWh (change if needed)
CO2_PER_KWH = 0.82      # kg CO2 per kWh (India grid approx.)


def true_energy(hour, temp, occ, app, weekend, rng=None):
    """Hidden 'real world' formula used only to create training data."""
    base = 0.25 + 0.12 * occ + 0.22 * app
    cooling = np.maximum(0, temp - 24) * 0.10 * (1 + 0.1 * occ)
    heating = np.maximum(0, 16 - temp) * 0.08
    evening = np.where((hour >= 18) & (hour <= 22), 0.9, 0.0)
    morning = np.where((hour >= 6) & (hour <= 9), 0.4, 0.0)
    night = np.where((hour >= 0) & (hour <= 5), -0.25, 0.0)
    day_home = np.where(weekend == 1, np.where((hour >= 10) & (hour <= 17), 0.5, 0.0), 0.0)
    energy = base + cooling + heating + evening + morning + night + day_home
    if rng is not None:
        energy = energy + rng.normal(0, 0.08, size=np.shape(energy))
    return np.maximum(energy, 0.05)


def build_dataset(n=6000, seed=42):
    rng = np.random.default_rng(seed)
    hour = rng.integers(0, 24, n)
    temp = rng.uniform(5, 45, n)
    occ = rng.integers(1, 9, n)
    app = rng.integers(0, 11, n)
    weekend = rng.integers(0, 2, n)
    X = np.column_stack([hour, temp, occ, app, weekend])
    y = true_energy(hour, temp, occ, app, weekend, rng)
    return X, y


def make_features(hour, temp, occ, app, weekend):
    """Turn raw inputs into model features (works for single values or arrays)."""
    hour, temp, occ, app, weekend = [np.asarray(v, dtype=float) for v in (hour, temp, occ, app, weekend)]
    cooling = np.maximum(0, temp - 24)
    heating = np.maximum(0, 16 - temp)
    evening = ((hour >= 18) & (hour <= 22)).astype(float)
    morning = ((hour >= 6) & (hour <= 9)).astype(float)
    night = (hour <= 5).astype(float)
    day_home = weekend * ((hour >= 10) & (hour <= 17)).astype(float)
    ones = np.ones_like(hour + temp + occ + app + weekend)
    return np.column_stack([ones, occ, app, cooling, cooling * occ, heating,
                            evening, morning, night, day_home])


class EnergyModel:
    """Linear regression (numpy only) trained on engineered features."""

    def __init__(self):
        X, y = build_dataset()
        idx = np.random.default_rng(1).permutation(len(y))
        cut = int(len(y) * 0.8)
        tr, te = idx[:cut], idx[cut:]
        F = make_features(*X.T)
        self.w, *_ = np.linalg.lstsq(F[tr], y[tr], rcond=None)
        pred = F[te] @ self.w
        ss_res = float(np.sum((y[te] - pred) ** 2))
        ss_tot = float(np.sum((y[te] - y[te].mean()) ** 2))
        self.r2 = round(1 - ss_res / ss_tot, 3)
        self.mae = round(float(np.mean(np.abs(y[te] - pred))), 3)

    def predict_one(self, hour, temp, occ, app, weekend):
        f = make_features([hour], [temp], [occ], [app], [weekend])
        return max(0.05, float((f @ self.w)[0]))

    def forecast_day(self, temp, occ, app, weekend):
        return [round(self.predict_one(h, temp, occ, app, weekend), 3) for h in range(24)]

    def optimized_day(self, temp, occ, app, weekend):
        """Scenario: run 1-2 heavy appliances outside peak (18-22h) and set AC smartly."""
        shift = 2 if app >= 4 else 1 if app >= 1 else 0
        eco_temp = temp - 2 if temp > 28 else temp   # AC at 24-26 acts like ~2 degrees less load
        out = []
        for h in range(24):
            a = app
            if 18 <= h <= 22:
                a = max(0, app - shift)
            elif 11 <= h <= 15:
                a = min(10, app + shift)  # moved to cheap solar hours
            out.append(self.predict_one(h, eco_temp, occ, a, weekend))
        return [round(v, 3) for v in out]


def recommendations(temp, occ, app, weekend, base, opt):
    tips = []
    peak_hour = int(np.argmax(base))
    tips.append(f"Peak usage is around {peak_hour:02d}:00. Avoid running washing machine, iron or geyser at that time.")
    if temp > 28:
        tips.append("Set AC to 24-26 C with fan on auto. Every 1 C higher saves about 6% cooling energy.")
    elif temp < 16:
        tips.append("Use room heaters only in occupied rooms and seal door gaps to reduce heating load.")
    else:
        tips.append("Weather is mild: open windows and use fans instead of AC/heater.")
    if app >= 5:
        tips.append("Many heavy appliances are running. Shift 1-2 of them to 11:00-15:00 (daytime/solar hours).")
    if occ >= 5:
        tips.append("High occupancy: use LED lighting and switch off devices in empty rooms.")
    if weekend:
        tips.append("Weekend detected: daytime load is higher, so plan laundry/cooking in one batch.")
    tips.append("Unplug standby devices (TV, chargers, set-top box) at night to cut idle power.")
    return tips
