from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
from agent.agent import FRISDIAgent
app = FastAPI(
    title="FRISDI AI",
    description="Fraud Risk Intelligence & Spike Detection Interface",
    version="1.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://frisdi.vercel.app"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained AI model
model_package = joblib.load("frisdi_model.pkl")

model = model_package["model"]
FEATURES = model_package["features"]

# Initialize FRISDI Agent
frisdi_agent = FRISDIAgent(
    model=model,
    features=FEATURES
)

class Transaction(BaseModel):
    amount: float
    hour: int
    account_age_days: int
    transactions_last_hour: int
    failed_transactions: int
    distance_from_home: float
    simulate_ml_failure: bool=False

@app.get("/")
def home():
    return {
        "system": "FRISDI AI",
        "status": "online"
    }


@app.get("/metrics")
def get_metrics():
    return {
        "precision": round(
            model_package["precision"] * 100,
            2
        ),
        "recall": round(
            model_package["recall"] * 100,
            2
        ),
        "false_positives": model_package["false_positives"],
        "false_positive_cost": model_package[
            "false_positive_cost"
        ],
        "test_size": model_package["test_size"]
    }


@app.post("/predict")
def predict(transaction: Transaction):

    transaction_data = pd.DataFrame(
        [[
            transaction.amount,
            transaction.hour,
            transaction.account_age_days,
            transaction.transactions_last_hour,
            transaction.failed_transactions,
            transaction.distance_from_home
        ]],
        columns=FEATURES
    )

    # Fraud probability from AI model
    fraud_probability = model.predict_proba(
        transaction_data
    )[0][1]

    # ===============================
    # FRISDI HYBRID RISK ENGINE
    # ===============================

    # Prevent extremely large transactions
    # from receiving an unrealistically low score
    if transaction.amount >= 100000:
        fraud_probability = max(
            fraud_probability,
            0.85
        )

    elif transaction.amount >= 50000:
        fraud_probability = max(
            fraud_probability,
            0.65
        )

    # Transaction velocity rules
    if transaction.transactions_last_hour >= 20:
        fraud_probability = max(
            fraud_probability,
            0.80
        )

    elif transaction.transactions_last_hour >= 10:
        fraud_probability = max(
            fraud_probability,
            0.60
        )

    # High amount + high velocity
    if (
        transaction.amount >= 50000
        and transaction.transactions_last_hour >= 10
    ):
        fraud_probability = max(
            fraud_probability,
            0.90
        )

    # Risk classification
    if fraud_probability >= 0.75:
        risk_level = "HIGH"

    elif fraud_probability >= 0.40:
        risk_level = "MEDIUM"

    else:
        risk_level = "LOW"

    # AI explanation
    reasons = []

    if transaction.amount > 15000:
        reasons.append(
            "Unusually high transaction amount"
        )

    if transaction.account_age_days < 30:
        reasons.append(
            "Very new account"
        )

    if transaction.transactions_last_hour > 6:
        reasons.append(
            "High transaction velocity"
        )

    if transaction.failed_transactions > 2:
        reasons.append(
            "Multiple failed transaction attempts"
        )

    if transaction.distance_from_home > 100:
        reasons.append(
            "Transaction unusually far from home"
        )

    if transaction.hour <= 4:
        reasons.append(
            "Unusual late-night activity"
        )

    if not reasons:
        reasons.append(
            "No major risk signals detected"
        )

    # Final response
    return {
        "risk_score": round(
            fraud_probability * 100,
            2
        ),
        "risk_level": risk_level,
        "reasons": reasons
    }
@app.post("/agent/investigate")
def investigate_transaction(transaction: Transaction):

    result = frisdi_agent.investigate(
        transaction
    )

    return result
