"""
ENSP355 - AI in Cyber Security | Lab 4
Explainable and Privacy-Aware AI Threat Alert

Student: Aditya | Roll Number: 2401010245

Run:
    pip install -r requirements.txt
    python lab4_explainable_privacy_alert.py

The script expects simple_cyber_activity.csv in the same directory.
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

BASE = Path(__file__).resolve().parent
DATA_FILE = BASE / "simple_cyber_activity.csv"
FEATURES = ["failed_logins", "login_hour", "new_device", "download_mb"]

# Load the small synthetic login-activity dataset.
data = pd.read_csv(DATA_FILE)
X = data[FEATURES]
y = data["label"]

# Retrain the Lab 2-style Decision Tree reproducibly.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)

print("Dataset shape:", data.shape)
print("Class counts (0=Normal, 1=Suspicious):")
print(y.value_counts().sort_index())
print(f"Decision Tree test accuracy: {model.score(X_test, y_test) * 100:.2f}%")
print("Confusion matrix [Normal, Suspicious]:")
print(__import__("sklearn.metrics").metrics.confusion_matrix(
    y_test, model.predict(X_test), labels=[0, 1]
))

# Q1. Five baseline predictions with confidence.
samples = pd.DataFrame([
    {"account_id": "ACC-1001", "ip_address": "192.168.10.24",
     "failed_logins": 0, "login_hour": 10, "new_device": 0, "download_mb": 40},
    {"account_id": "ACC-1002", "ip_address": "10.0.4.18",
     "failed_logins": 7, "login_hour": 2, "new_device": 1, "download_mb": 150},
    {"account_id": "ACC-1003", "ip_address": "172.16.8.90",
     "failed_logins": 1, "login_hour": 15, "new_device": 0, "download_mb": 720},
    {"account_id": "ACC-1004", "ip_address": "192.168.20.11",
     "failed_logins": 4, "login_hour": 23, "new_device": 1, "download_mb": 400},
    {"account_id": "ACC-1005", "ip_address": "10.1.1.9",
     "failed_logins": 2, "login_hour": 9, "new_device": 0, "download_mb": 80},
])
samples["prediction"] = [
    "Suspicious" if p == 1 else "Normal" for p in model.predict(samples[FEATURES])
]
samples["confidence"] = model.predict_proba(samples[FEATURES]).max(axis=1).round(3)
print("\nQ1 - Baseline predictions:")
print(samples.to_string(index=False))
lowest = samples.loc[samples["confidence"].idxmin(), "account_id"]
print("Lowest-confidence record(s):",
      ", ".join(samples.loc[samples["confidence"] == samples["confidence"].min(),
                            "account_id"].tolist()))

# Q2. Feature importance and plain-language explanation.
importance = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)
print("\nQ2 - Feature importance:")
print(importance.to_string(index=False))
plt.figure(figsize=(7, 4))
plt.bar(importance["feature"], importance["importance"])
plt.title("Feature Importance for Login Alert")
plt.ylabel("Importance")
plt.xticks(rotation=25)
plt.tight_layout()
plt.show()

print("\nTop-three feature explanations:")
explanations = {
    "download_mb": "Download volume was the most influential feature in this fitted tree.",
    "failed_logins": "The number of failed login attempts was the second most influential feature.",
    "login_hour": "Login hour had zero importance in this fitted tree; it was not used for its splits.",
    "new_device": "New-device status had zero importance in this fitted tree; it was not used for its splits."
}
for feature in importance.head(3)["feature"]:
    print(f"- {feature}: {explanations[feature]}")

# Q3. Controlled sensitivity test: change only failed_logins from 2 to 8.
baseline = samples.iloc[[4]].copy()
changed = baseline.copy()
changed["failed_logins"] = 8
baseline_prediction = model.predict(baseline[FEATURES])[0]
baseline_confidence = model.predict_proba(baseline[FEATURES]).max()
changed_prediction = model.predict(changed[FEATURES])[0]
changed_confidence = model.predict_proba(changed[FEATURES]).max()
print("\nQ3 - Controlled sensitivity:")
print(f"Baseline: {'Suspicious' if baseline_prediction else 'Normal'} "
      f"(confidence {baseline_confidence:.3f})")
print(f"Changed : {'Suspicious' if changed_prediction else 'Normal'} "
      f"(confidence {changed_confidence:.3f})")
print("Only failed_logins changed: 2 -> 8.")

# Q4. Privacy masking and responsible-use rules.
def mask_ip(ip):
    parts = str(ip).split(".")
    if len(parts) == 4:
        return f"{parts[0]}.{parts[1]}.xxx.xxx"
    return "<MASKED_IP>"

def mask_account(account_id):
    text = str(account_id)
    return text[:4] + "***" + text[-2:]

samples["masked_ip"] = samples["ip_address"].apply(mask_ip)
samples["masked_account"] = samples["account_id"].apply(mask_account)
print("\nQ4 - Masked output (raw identifiers excluded):")
print(samples[["masked_account", "masked_ip", "prediction", "confidence"]].to_string(index=False))

privacy_rules = [
    "Mask identifiers before display or export.",
    "Use sanitized data in classroom notebooks.",
    "Restrict access to unmasked records.",
    "Require human review for high-risk action."
]
print("\nPrivacy and responsible-use rules:")
for rule in privacy_rules:
    print("-", rule)

# Q5. Responsible alert generation.
def numeric_reason(row):
    reasons = []
    if row["failed_logins"] >= 5:
        reasons.append("many failed logins")
    if row["new_device"] == 1:
        reasons.append("new device")
    if row["login_hour"] < 6 or row["login_hour"] > 22:
        reasons.append("unusual login time")
    if row["download_mb"] >= 500:
        reasons.append("large data download")
    return ", ".join(reasons) if reasons else "no strong warning"

def make_alert(row):
    reason = numeric_reason(row)
    if row["prediction"] == "Suspicious":
        severity = "High"
        action = "Verify, contain if required, and report."
    elif row["confidence"] < 0.70:
        severity = "Medium"
        action = "Send for analyst review before action."
    else:
        severity = "Low"
        action = "Continue normal monitoring."
    human_review = "Yes" if severity in ["High", "Medium"] else "No"
    return pd.Series([severity, reason, action, human_review],
                     index=["severity", "reason", "recommended_action", "human_review"])

alerts = samples.apply(make_alert, axis=1)
final_alerts = pd.concat([samples, alerts], axis=1)
print("\nQ5 - Final responsible alert table:")
print(final_alerts[[
    "masked_account", "prediction", "confidence", "severity",
    "reason", "recommended_action", "human_review"
]].to_string(index=False))

print("\nLimitation: this is a small synthetic classroom dataset. A confident prediction "
      "is not proof of malicious activity; alerts require human verification.")
