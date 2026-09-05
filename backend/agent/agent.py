from agent.tools import (
    ml_risk_analysis_with_failure,
    rule_analysis,
    behaviour_analysis,
    account_analysis
)

from agent.state import InvestigationState


class FRISDIAgent:

    def __init__(self, model, features):
        self.model = model
        self.features = features

    # ==================================================
    # EVIDENCE-AWARE PLANNER
    # ==================================================

    def choose_next_action(self, state, transaction):
        """
        Chooses the next investigation action based on:

        1. Initial transaction signals
        2. Evidence already collected
        3. Missing evidence
        4. Current uncertainty
        """

        candidates = []

        evidence = state.evidence

        # ==================================================
        # ML ANALYZER
        # ==================================================

        if "ml" not in evidence:

            ml_priority = 20

            # ML is useful when we need an independent
            # overall fraud estimate.
            ml_reason = (
                "An independent ML risk estimate "
                "is still unavailable."
            )

            if "rules" in evidence:
                ml_priority += 20
                ml_reason = (
                    "Rules have produced evidence, but "
                    "an independent ML estimate is needed "
                    "to corroborate the findings."
                )

            if "behaviour" in evidence:
                ml_priority += 20
                ml_reason = (
                    "Behavioural anomalies have been "
                    "identified, but an independent ML "
                    "estimate is needed to corroborate them."
                )

            if transaction.amount >= 50000:
                ml_priority += 10

            candidates.append(
                ("ml", ml_priority, ml_reason)
            )

        # ==================================================
        # RULE ANALYZER
        # ==================================================

        if "rules" not in evidence:

            rule_priority = 20

            rule_reasons = []

            if transaction.amount >= 50000:
                rule_priority += 25
                rule_reasons.append(
                    "high transaction amount"
                )

            if transaction.transactions_last_hour >= 10:
                rule_priority += 25
                rule_reasons.append(
                    "high transaction velocity"
                )

            if transaction.failed_transactions > 2:
                rule_priority += 15
                rule_reasons.append(
                    "multiple failed attempts"
                )

            if rule_reasons:

                rule_reason = (
                    "Explicit rules can verify "
                    + ", ".join(rule_reasons)
                    + "."
                )

            else:

                rule_reason = (
                    "No rule-based evidence has been "
                    "collected yet."
                )

            # If behaviour has already been checked,
            # rules become more useful as an independent
            # evidence source.
            if "behaviour" in evidence:
                rule_priority += 15
                rule_reason += (
                    " Behavioural evidence exists, so "
                    "rule-based corroboration would "
                    "strengthen the investigation."
                )

            candidates.append(
                ("rules", rule_priority, rule_reason)
            )

        # ==================================================
        # BEHAVIOUR ANALYZER
        # ==================================================

        if "behaviour" not in evidence:

            behaviour_priority = 20

            behaviour_reasons = []

            if transaction.distance_from_home > 100:
                behaviour_priority += 25
                behaviour_reasons.append(
                    "unusual geographic distance"
                )

            if transaction.hour <= 4:
                behaviour_priority += 20
                behaviour_reasons.append(
                    "late-night activity"
                )

            if transaction.transactions_last_hour > 6:
                behaviour_priority += 15
                behaviour_reasons.append(
                    "elevated transaction velocity"
                )

            if behaviour_reasons:

                behaviour_reason = (
                    "Behavioural analysis can investigate "
                    + ", ".join(behaviour_reasons)
                    + "."
                )

            else:

                behaviour_reason = (
                    "No behavioural analysis has been "
                    "performed yet."
                )

            # If rules already found strong evidence,
            # behaviour provides an independent dimension.
            if "rules" in evidence:
                behaviour_priority += 20
                behaviour_reason += (
                    " Rule evidence is already available, "
                    "so behavioural analysis can provide "
                    "an independent perspective."
                )

            candidates.append(
                (
                    "behaviour",
                    behaviour_priority,
                    behaviour_reason
                )
            )

        # ==================================================
        # ACCOUNT ANALYZER
        # ==================================================

        if "account" not in evidence:

            account_priority = 15

            if transaction.account_age_days < 30:
                account_priority += 35

                account_reason = (
                    "The account is very new, so account "
                    "characteristics may provide important "
                    "context."
                )

            else:

                account_reason = (
                    "Account context has not yet been "
                    "investigated."
                )

            # Account evidence becomes more useful when
            # other signals are already suspicious.
            if len(evidence) >= 2:
                account_priority += 10
                account_reason += (
                    " Multiple suspicious signals already "
                    "exist, so account context can help "
                    "validate the overall assessment."
                )

            candidates.append(
                (
                    "account",
                    account_priority,
                    account_reason
                )
            )

        # ==================================================
        # NO MORE TOOLS
        # ==================================================

        if not candidates:

            return (
                None,
                "All available investigation tools "
                "have already been used."
            )

        # ==================================================
        # SORT BY INVESTIGATION VALUE
        # ==================================================

        candidates.sort(
            key=lambda item: item[1],
            reverse=True
        )

        next_action, priority, reason = candidates[0]

        thought = (
            f"I evaluated the current evidence and "
            f"remaining investigation options. "
            f"The best next action is '{next_action}' "
            f"with priority {priority}. "
            f"Reason: {reason}"
        )

        return next_action, thought

    # ==================================================
    # MAIN INVESTIGATION
    # ==================================================

    def investigate(self, transaction):

        state = InvestigationState(transaction)

        # ==================================================
        # STEP 1: OBSERVE
        # ==================================================

        if transaction.amount >= 50000:
            state.observations.append(
                "High-value transaction detected"
            )

        if transaction.transactions_last_hour >= 10:
            state.observations.append(
                "High transaction velocity detected"
            )

        if transaction.failed_transactions > 2:
            state.observations.append(
                "Multiple failed attempts detected"
            )

        if transaction.distance_from_home > 100:
            state.observations.append(
                "Geographic anomaly detected"
            )

        if transaction.hour <= 4:
            state.observations.append(
                "Late-night transaction detected"
            )

        if transaction.account_age_days < 30:
            state.observations.append(
                "New account detected"
            )

        if not state.observations:
            state.observations.append(
                "No obvious anomalies detected"
            )

        # ==================================================
        # STEP 2: INITIAL REASONING
        # ==================================================

        step = 1

        state.log(
            step=step,
            thought=(
                "I need to determine whether this "
                "transaction is safe, suspicious, "
                "or requires intervention."
            ),
            observation=state.observations
        )

        step += 1

        # ==================================================
        # STEP 3: INITIAL PLAN
        # ==================================================

        next_action, thought = self.choose_next_action(
            state,
            transaction
        )

        state.log(
            step=step,
            thought=thought,
            action=next_action
        )

        step += 1

        # ==================================================
        # STEP 4: AGENTIC INVESTIGATION LOOP
        # ==================================================

        investigated = set()

        max_steps = 6

        while (
            next_action is not None
            and len(investigated) < max_steps
        ):

            # ==================================================
            # ML RISK ANALYZER
            # ==================================================

            if next_action == "ml":

                try:

                    result = ml_risk_analysis_with_failure(
                        self.model,
                        self.features,
                        transaction,
                        simulate_failure=(
                            transaction.simulate_ml_failure
                        )
                    )

                    state.evidence["ml"] = result

                    state.actions_taken.append(
                        "ML Risk Analyzer"
                    )

                    investigated.add("ml")

                    state.log(
                        step=step,
                        thought=(
                            "The ML model provides an "
                            "independent fraud probability. "
                            "I will compare it with the "
                            "other evidence."
                        ),
                        action="ML Risk Analyzer",
                        observation=result
                    )

                except Exception as error:

                    # ==========================================
                    # FAILURE DETECTED
                    # ==========================================

                    state.adaptation = {
                        "trigger": (
                            "ML Risk Analyzer failed"
                        ),
                        "error": str(error),
                        "response": (
                            "Replanned investigation"
                        )
                    }

                    state.log(
                        step=step,
                        thought=(
                            "The ML analyzer failed. "
                            "Its evidence is unavailable, "
                            "so I will reassess the "
                            "remaining independent tools."
                        ),
                        action="REPLAN",
                        observation=str(error)
                    )

                    investigated.add("ml")

                    # ==========================================
                    # ADAPTIVE REPLANNING
                    # ==========================================

                    next_action, fallback_thought = (
                        self.choose_next_action(
                            state,
                            transaction
                        )
                    )

                    state.log(
                        step=step + 1,
                        thought=(
                            "The previous tool failed. "
                            "I recalculated the available "
                            "actions and selected a fallback "
                            "investigation path."
                        ),
                        action=next_action
                    )

                    step += 2

                    continue

            # ==================================================
            # RULE ANALYZER
            # ==================================================

            elif next_action == "rules":

                result = rule_analysis(
                    transaction
                )

                state.evidence["rules"] = result

                state.actions_taken.append(
                    "Rule Analyzer"
                )

                investigated.add("rules")

                state.log(
                    step=step,
                    thought=(
                        "I will compare the transaction "
                        "against explicit fraud-risk rules "
                        "to identify concrete violations."
                    ),
                    action="Rule Analyzer",
                    observation=result
                )

                state.reasons.extend(
                    result["rules_triggered"]
                )

            # ==================================================
            # BEHAVIOUR ANALYZER
            # ==================================================

            elif next_action == "behaviour":

                result = behaviour_analysis(
                    transaction
                )

                state.evidence["behaviour"] = result

                state.actions_taken.append(
                    "Behaviour Analyzer"
                )

                investigated.add("behaviour")

                state.log(
                    step=step,
                    thought=(
                        "I will investigate behavioural "
                        "patterns to determine whether "
                        "the transaction deviates from "
                        "normal activity."
                    ),
                    action="Behaviour Analyzer",
                    observation=result
                )

                state.reasons.extend(
                    result["anomalies"]
                )

            # ==================================================
            # ACCOUNT ANALYZER
            # ==================================================

            elif next_action == "account":

                result = account_analysis(
                    transaction
                )

                state.evidence["account"] = result

                state.actions_taken.append(
                    "Account Analyzer"
                )

                investigated.add("account")

                state.log(
                    step=step,
                    thought=(
                        "I will examine account context "
                        "to determine whether the account "
                        "characteristics increase the risk."
                    ),
                    action="Account Analyzer",
                    observation=result
                )

                state.reasons.extend(
                    result["observations"]
                )

            step += 1

            # ==================================================
            # STEP 5: EVALUATE CURRENT EVIDENCE
            # ==================================================

            ml_score = 0

            if "ml" in state.evidence:

                ml_score = state.evidence[
                    "ml"
                ]["fraud_probability"]

            rule_count = 0

            if "rules" in state.evidence:

                rule_count = state.evidence[
                    "rules"
                ]["rule_count"]

            anomaly_count = 0

            if "behaviour" in state.evidence:

                anomaly_count = state.evidence[
                    "behaviour"
                ]["anomaly_count"]

            # ------------------------------------------
            # Start with ML evidence
            # ------------------------------------------

            state.risk_score = ml_score

            # ------------------------------------------
            # Rule evidence
            # ------------------------------------------

            if rule_count >= 3:

                state.risk_score = max(
                    state.risk_score,
                    90
                )

            elif rule_count == 2:

                state.risk_score = max(
                    state.risk_score,
                    75
                )

            elif rule_count == 1:

                state.risk_score = max(
                    state.risk_score,
                    55
                )

            # ------------------------------------------
            # Behaviour evidence
            # ------------------------------------------

            if anomaly_count >= 2:

                state.risk_score = max(
                    state.risk_score,
                    80
                )

            elif anomaly_count == 1:

                state.risk_score = max(
                    state.risk_score,
                    55
                )

            # ==================================================
            # STEP 6: EVIDENCE-AWARE REPLANNING
            # ==================================================

            next_action, thought = self.choose_next_action(
                state,
                transaction
            )

            # ------------------------------------------
            # Strong evidence condition
            # ------------------------------------------

            if (
                state.risk_score >= 75
                and len(state.evidence) >= 2
            ):

                next_action = None

                state.log(
                    step=step,
                    thought=(
                        "The current evidence is strong "
                        "enough to support a high-confidence "
                        "decision. I will stop the "
                        "investigation rather than perform "
                        "unnecessary analysis."
                    ),
                    action="STOP"
                )

            elif next_action is not None:

                state.log(
                    step=step,
                    thought=(
                        "I reassessed the investigation "
                        "after receiving new evidence. "
                        "I will investigate the remaining "
                        "uncertainty using the next "
                        "highest-value tool."
                    ),
                    action=next_action
                )

            step += 1

        # ==================================================
        # STEP 7: FINAL EVALUATION
        # ==================================================

        evidence_sources = len(
            state.evidence
        )

        if evidence_sources >= 2:

            state.confidence = min(
                99,
                75 + evidence_sources * 5
            )

        else:

            state.confidence = 60

        # ==================================================
        # STEP 8: FINAL DECISION
        # ==================================================

        if state.risk_score >= 75:

            state.risk_level = "HIGH"
            state.decision = "BLOCK"

        elif state.risk_score >= 40:

            state.risk_level = "MEDIUM"
            state.decision = "REVIEW"

        else:

            state.risk_level = "LOW"
            state.decision = "ALLOW"

        if not state.reasons:

            state.reasons.append(
                "No major risk signals detected"
            )

        state.completed = True

        # ==================================================
        # FINAL RESPONSE
        # ==================================================

        return {
            "goal": (
                "Determine whether the transaction "
                "should be allowed, reviewed, or blocked."
            ),

            "observations":
                state.observations,

            "investigation_log":
                state.investigation_log,

            "actions_taken":
                state.actions_taken,

            "adaptation":
                state.adaptation,

            "evidence":
                state.evidence,

            "risk_score":
                round(
                    state.risk_score,
                    2
                ),

            "risk_level":
                state.risk_level,

            "confidence":
                state.confidence,

            "decision":
                state.decision,

            "reasons":
                list(
                    dict.fromkeys(
                        state.reasons
                    )
                )
        }
