import pandas as pd


def ml_risk_analysis(model, features, transaction):
    """
    Runs FRISDI's trained ML model.
    """

    transaction_data = pd.DataFrame(
        [[
            transaction.amount,
            transaction.hour,
            transaction.account_age_days,
            transaction.transactions_last_hour,
            transaction.failed_transactions,
            transaction.distance_from_home
        ]],
        columns=features
    )

    probability = model.predict_proba(
        transaction_data
    )[0][1]

    return {
        "tool": "ML Risk Analyzer",
        "fraud_probability": round(probability * 100, 2)
    }


def rule_analysis(transaction):
    """
    Checks explicit fraud-risk rules.
    """

    triggered = []

    if transaction.amount >= 100000:
        triggered.append(
            "Extremely high transaction amount"
        )

    elif transaction.amount >= 50000:
        triggered.append(
            "High transaction amount"
        )

    if transaction.transactions_last_hour >= 20:
        triggered.append(
            "Very high transaction velocity"
        )

    elif transaction.transactions_last_hour >= 10:
        triggered.append(
            "High transaction velocity"
        )

    if transaction.failed_transactions > 2:
        triggered.append(
            "Multiple failed transaction attempts"
        )

    if transaction.account_age_days < 30:
        triggered.append(
            "Very new account"
        )

    return {
        "tool": "Rule Analyzer",
        "rules_triggered": triggered,
        "rule_count": len(triggered)
    }


def behaviour_analysis(transaction):
    """
    Looks for behavioural anomalies.
    """

    anomalies = []

    if transaction.distance_from_home > 100:
        anomalies.append(
            "Unusual geographic location"
        )

    if transaction.hour <= 4:
        anomalies.append(
            "Unusual late-night activity"
        )

    if transaction.transactions_last_hour > 6:
        anomalies.append(
            "Unusual transaction velocity"
        )

    return {
        "tool": "Behaviour Analyzer",
        "anomalies": anomalies,
        "anomaly_count": len(anomalies)
    }


def account_analysis(transaction):
    """
    Examines account characteristics.
    """

    observations = []

    if transaction.account_age_days < 30:
        observations.append(
            "Account is recently created"
        )
    else:
        observations.append(
            "Account is not recently created"
        )

    return {
        "tool": "Account Analyzer",
        "observations": observations
    }
def ml_risk_analysis_with_failure(
    model,
    features,
    transaction,
    simulate_failure=False
):
    """
    ML analysis tool with optional failure simulation.

    Used to demonstrate FRISDI's ability to recover
    when an analysis tool becomes unavailable.
    """

    if simulate_failure:
        raise RuntimeError(
            "ML Risk Analyzer is temporarily unavailable"
        )

    return ml_risk_analysis(
        model,
        features,
        transaction
    )
