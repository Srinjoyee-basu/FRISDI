const API_URL = "https://frisdi.onrender.com";

const form = document.getElementById("transaction-form");
const analyzeButton = document.querySelector(".analyze-button");

const precisionElement = document.getElementById("precision");
const recallElement = document.getElementById("recall");
const falsePositivesElement =
    document.getElementById("false-positives");
const testSizeElement = document.getElementById("test-size");

const emptyResult = document.getElementById("empty-result");
const result = document.getElementById("result");

const riskScoreElement = document.getElementById("risk-score");
const riskLevelElement = document.getElementById("risk-level");
const scoreCircle = document.getElementById("score-circle");
const reasonsList = document.getElementById("reasons-list");


// ===============================
// GET TRANSACTION DATA
// ===============================

function getTransactionData() {
    return {
        amount: Number(
            document.getElementById("amount").value
        ),

        hour: Number(
            document.getElementById("hour").value
        ),

        account_age_days: Number(
            document.getElementById("account-age").value
        ),

        transactions_last_hour: Number(
            document.getElementById("velocity").value
        ),

        failed_transactions: Number(
            document.getElementById("failed").value
        ),

        distance_from_home: Number(
            document.getElementById("distance").value
        )
    };
}


// ===============================
// LOAD MODEL METRICS
// ===============================

async function loadMetrics() {

    try {
        const response = await fetch(
            `${API_URL}/metrics`
        );

        if (!response.ok) {
            throw new Error(
                "Could not load model metrics"
            );
        }

        const data = await response.json();

        precisionElement.textContent =
            `${data.precision}%`;

        recallElement.textContent =
            `${data.recall}%`;

        falsePositivesElement.textContent =
            data.false_positives;

        testSizeElement.textContent =
            data.test_size;

    } catch (error) {

        console.error(
            "Metrics error:",
            error
        );

        precisionElement.textContent = "Offline";
        recallElement.textContent = "Offline";
        falsePositivesElement.textContent = "--";
        testSizeElement.textContent = "--";
    }
}


// ===============================
// LIVE AI SIMULATION
// ===============================

let simulatorTimeout = null;

const simulatorInputs = [
    document.getElementById("amount"),
    document.getElementById("hour"),
    document.getElementById("account-age"),
    document.getElementById("velocity"),
    document.getElementById("failed"),
    document.getElementById("distance")
];


async function runRiskSimulation() {

    const transaction = getTransactionData();

    try {

        const response = await fetch(
            `${API_URL}/predict`,
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify(transaction)
            }
        );

        if (!response.ok) {
            throw new Error(
                "Prediction failed"
            );
        }

        const data = await response.json();

        displayResult(data);

    } catch (error) {

        console.error(
            "FRISDI simulator error:",
            error
        );
    }
}


// ===============================
// WATCH INPUT CHANGES
// ===============================

simulatorInputs.forEach((input) => {

    input.addEventListener("input", () => {

        clearTimeout(simulatorTimeout);

        simulatorTimeout = setTimeout(
            runRiskSimulation,
            500
        );

    });

});


// ===============================
// MANUAL SIMULATION BUTTON
// ===============================

form.addEventListener(
    "submit",
    async (event) => {

        event.preventDefault();

        analyzeButton.disabled = true;

        analyzeButton.textContent =
            "FRISDI AI ANALYZING...";

        await runRiskSimulation();

        analyzeButton.disabled = false;

        analyzeButton.textContent =
            "Run FRISDI Simulation";
    }
);


// ===============================
// DISPLAY AI RESULT
// ===============================

function displayResult(data) {

    emptyResult.classList.add("hidden");

    result.classList.remove("hidden");

    riskScoreElement.textContent =
        `${data.risk_score}%`;

    riskLevelElement.textContent =
        `${data.risk_level} RISK`;

    scoreCircle.classList.remove(
        "low",
        "medium",
        "high"
    );

    const level =
        data.risk_level.toLowerCase();

    scoreCircle.classList.add(level);

    reasonsList.innerHTML = "";

    data.reasons.forEach((reason) => {

        const item =
            document.createElement("li");

        item.textContent = reason;

        reasonsList.appendChild(item);

    });

}


// ===============================
// START FRISDI
// ===============================

loadMetrics();
// ===============================
// SIDEBAR NAVIGATION
// ===============================

const navItems = document.querySelectorAll(".nav-item");

navItems.forEach((item) => {

    item.addEventListener("click", () => {

        navItems.forEach((nav) => {
            nav.classList.remove("active");
        });

        item.classList.add("active");

        const text = item.textContent.trim();

        if (text.includes("LIVE AI RISK SIMULATOR")) {

            document
                .querySelector(".analysis-panel")
                .scrollIntoView({
                    behavior: "smooth",
                    block: "center"
                });

        }

        if (text.includes("Dashboard")) {

            window.scrollTo({
                top: 0,
                behavior: "smooth"
            });

        }

        if (text.includes("Simulate Transaction Risk")) {

            document
                .querySelector(".analysis-panel")
                .scrollIntoView({
                    behavior: "smooth",
                    block: "center"
                });

            document
                .getElementById("amount")
                .focus();

        }

    });

});
