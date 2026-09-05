import os
from openai import OpenAI


def analyze_transaction(transaction_data):
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY environment variable is not set"
        )

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )

    prompt = f"""
You are FRISDI, an AI-powered fraud investigation analyst.

Analyze the transaction and the evidence collected by FRISDI's
fraud detection system.

IMPORTANT:
- Do not invent evidence.
- Only use the information provided.
- Explain the strongest risk signals.
- Explain why the current risk level makes sense.
- Give a concise professional investigation summary.
- Do not change the deterministic risk score or decision.

Transaction and investigation data:

{transaction_data}

Return your response with these sections:

INVESTIGATION SUMMARY:
KEY FINDINGS:
RISK ASSESSMENT:
RECOMMENDED ACTION:
"""

    response = client.chat.completions.create(
        model="openrouter/free",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are FRISDI's fraud investigation "
                    "analyst. Be factual and concise."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content
