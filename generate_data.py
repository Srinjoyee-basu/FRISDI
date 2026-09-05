import pandas as pd
import numpy as np
import random

np.random.seed(42)
random.seed(42)

NUM_TRANSACTIONS = 20000

transactions = []

for i in range(NUM_TRANSACTIONS):

    amount = round(np.random.exponential(3000), 2)

    hour = random.randint(0, 23)

    account_age_days = random.randint(1, 3000)

    transactions_last_hour = np.random.poisson(2)

    failed_transactions = np.random.poisson(0.5)

    distance_from_home = round(
        abs(np.random.normal(15, 20)),
        2
    )

    fraud_probability = 0.02

    # High transaction amount
    if amount > 15000:
        fraud_probability += 0.15

    # Very new account
    if account_age_days < 30:
        fraud_probability += 0.20

    # Unusual transaction velocity
    if transactions_last_hour > 6:
        fraud_probability += 0.25

    # Multiple failed attempts
    if failed_transactions > 2:
        fraud_probability += 0.20

    # Transaction far from home
    if distance_from_home > 100:
        fraud_probability += 0.15

    # Late-night activity
    if hour <= 4:
        fraud_probability += 0.08

    fraud_probability = min(fraud_probability, 0.95)

    is_fraud = np.random.random() < fraud_probability

    transactions.append({
        "transaction_id": i + 1,
        "amount": amount,
        "hour": hour,
        "account_age_days": account_age_days,
        "transactions_last_hour": transactions_last_hour,
        "failed_transactions": failed_transactions,
        "distance_from_home": distance_from_home,
        "is_fraud": int(is_fraud)
    })


df = pd.DataFrame(transactions)

df.to_csv(
    "data/transactions.csv",
    index=False
)

print("\nDataset created successfully!")
print("------------------------------")
print(f"Total transactions: {len(df)}")
print(f"Fraud transactions: {df['is_fraud'].sum()}")
print(f"Fraud rate: {df['is_fraud'].mean() * 100:.2f}%")
print("\nSaved to: data/transactions.csv")
