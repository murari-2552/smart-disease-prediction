# ============================================================
# SMART DISEASE PREDICTION SYSTEM
# Professional Gradio UI
# ============================================================

import gradio as gr
import pandas as pd
import numpy as np
import joblib


# ============================================================
# LOAD TRAINED MODELS
# ============================================================

MODEL_PATH = "data/models/ensemble_model (1).pkl"
LABEL_ENCODER_PATH = "data/models/label_encoder.pkl"
SYMPTOM_COLUMNS_PATH = "data/models/symptom_columns.pkl"


print("Loading trained models...")

ensemble_model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)
symptom_columns = joblib.load(SYMPTOM_COLUMNS_PATH)

print("Models loaded successfully!")
print("Number of symptoms:", len(symptom_columns))


# ============================================================
# EXTRACT INDIVIDUAL MODELS
# ============================================================

lr_model = ensemble_model.estimators_[0]
rf_model = ensemble_model.estimators_[1]
svm_model = ensemble_model.estimators_[2]


# ============================================================
# SYMPTOM DISPLAY NAMES
# ============================================================

def format_symptom(symptom):
    """
    Convert dataset-style symptom names into readable names.
    Example:
    muscle_pain -> Muscle Pain
    mild_fever -> Mild Fever
    """
    return symptom.replace("_", " ").title()


symptom_choices = [
    (format_symptom(symptom), symptom)
    for symptom in symptom_columns
]


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_disease(selected_symptoms):

    # --------------------------------------------------------
    # Check if symptoms were selected
    # --------------------------------------------------------

    if not selected_symptoms:
        empty_table = pd.DataFrame(
            [["-", "-", "-", "-"]],
            columns=[
                "Model",
                "Prediction",
                "Status",
                "Confidence"
            ]
        )

        return (
            """
            <div class="empty-result">
                <div class="empty-icon">🩺</div>
                <h2>No symptoms selected</h2>
                <p>Please select at least one symptom to generate a prediction.</p>
            </div>
            """,
            "—",
            "0.00%",
            pd.DataFrame(
                [
                    ["1", "—", "0.00%"],
                    ["2", "—", "0.00%"],
                    ["3", "—", "0.00%"]
                ],
                columns=["Rank", "Disease", "Confidence"]
            ),
            empty_table
        )


    # --------------------------------------------------------
    # Create input dataframe
    # --------------------------------------------------------

    input_data = pd.DataFrame(
        0,
        index=[0],
        columns=symptom_columns
    )


    # --------------------------------------------------------
    # Activate selected symptoms
    # --------------------------------------------------------

    for symptom in selected_symptoms:

        # Dropdown returns the original value
        if symptom in input_data.columns:
            input_data.loc[0, symptom] = 1


    # ========================================================
    # INDIVIDUAL MODEL PREDICTIONS
    # ========================================================

    lr_prediction = lr_model.predict(input_data)[0]
    rf_prediction = rf_model.predict(input_data)[0]
    svm_prediction = svm_model.predict(input_data)[0]


    # Convert encoded labels to disease names
    lr_disease = label_encoder.inverse_transform(
        [int(lr_prediction)]
    )[0]

    rf_disease = label_encoder.inverse_transform(
        [int(rf_prediction)]
    )[0]

    svm_disease = label_encoder.inverse_transform(
        [int(svm_prediction)]
    )[0]


    # ========================================================
    # SOFT VOTING ENSEMBLE
    # ========================================================

    probabilities = ensemble_model.predict_proba(
        input_data
    )[0]

    class_indices = ensemble_model.classes_

    # Sort probabilities from highest to lowest
    top_positions = np.argsort(probabilities)[::-1][:3]


    # Best prediction
    best_position = top_positions[0]

    predicted_index = class_indices[best_position]

    predicted_disease = label_encoder.inverse_transform(
        [int(predicted_index)]
    )[0]

    confidence = probabilities[best_position] * 100


    # ========================================================
    # TOP 3 PREDICTIONS
    # ========================================================

    top_predictions = []

    for rank, position in enumerate(top_positions, start=1):

        class_index = class_indices[position]

        disease = label_encoder.inverse_transform(
            [int(class_index)]
        )[0]

        probability = probabilities[position] * 100

        top_predictions.append([
            rank,
            disease,
            f"{probability:.2f}%"
        ])


    top3_df = pd.DataFrame(
        top_predictions,
        columns=[
            "Rank",
            "Disease",
            "Confidence"
        ]
    )


    # ========================================================
    # MODEL COMPARISON
    # ========================================================

    comparison_df = pd.DataFrame(
        [
            [
                "Logistic Regression",
                lr_disease,
                "✓ Correct / Prediction",
                "—"
            ],
            [
                "Random Forest",
                rf_disease,
                "✓ Correct / Prediction",
                "—"
            ],
            [
                "SVM",
                svm_disease,
                "✓ Correct / Prediction",
                "—"
            ],
            [
                "Soft Voting Ensemble",
                predicted_disease,
                "★ Final Prediction",
                f"{confidence:.2f}%"
            ]
        ],
        columns=[
            "Model",
            "Prediction",
            "Status",
            "Confidence"
        ]
    )


    # ========================================================
    # CONFIDENCE LEVEL
    # ========================================================

    if confidence >= 80:
        confidence_level = "High"
    elif confidence >= 60:
        confidence_level = "Moderate"
    else:
        confidence_level = "Low"


    # ========================================================
    # RESULT CARD
    # ========================================================

    result_html = f"""
    <div class="result-card">

        <div class="result-header">
            <div class="result-icon">🩺</div>

            <div>
                <div class="result-label">
                    PREDICTED DISEASE
                </div>

                <div class="result-disease">
                    {predicted_disease}
                </div>
            </div>
        </div>


        <div class="confidence-section">

            <div class="confidence-top">

                <span>Model Confidence</span>

                <strong>
                    {confidence:.2f}%
                </strong>

            </div>


            <div class="confidence-bar">

                <div
                    class="confidence-fill"
                    style="width:{min(confidence, 100):.2f}%"
                ></div>

            </div>


            <div class="confidence-bottom">
                Confidence level: <strong>{confidence_level}</strong>
            </div>

        </div>

    </div>
    """


    return (
        result_html,
        predicted_disease,
        f"{confidence:.2f}%",
        top3_df,
        comparison_df
    )


# ============================================================
# CLEAR FUNCTION
# ============================================================

def clear_results():

    empty_table = pd.DataFrame(
        [
            ["1", "—", "0.00%"],
            ["2", "—", "0.00%"],
            ["3", "—", "0.00%"]
        ],
        columns=[
            "Rank",
            "Disease",
            "Confidence"
        ]
    )

    comparison_table = pd.DataFrame(
        [
            ["Logistic Regression", "—", "—", "—"],
            ["Random Forest", "—", "—", "—"],
            ["SVM", "—", "—", "—"],
            ["Soft Voting Ensemble", "—", "—", "—"]
        ],
        columns=[
            "Model",
            "Prediction",
            "Status",
            "Confidence"
        ]
    )

    return (
        [],
        """
        <div class="empty-result">
            <div class="empty-icon">🩺</div>
            <h2>Ready for prediction</h2>
            <p>Select your symptoms and click <b>Predict Disease</b>.</p>
        </div>
        """,
        "—",
        "0.00%",
        empty_table,
        comparison_table
    )


# ============================================================
# CUSTOM CSS
# ============================================================

custom_css = """

/* ----------------------------------------------------------
   MAIN APPLICATION
---------------------------------------------------------- */

.gradio-container {
    max-width: 1250px !important;
    margin: auto !important;
    font-family: Inter, Arial, sans-serif !important;
}


/* ----------------------------------------------------------
   HEADER
---------------------------------------------------------- */

.app-header {
    text-align: center;
    padding: 28px 20px 20px 20px;
    margin-bottom: 15px;
}

.app-title {
    font-size: 38px;
    font-weight: 800;
    margin-bottom: 8px;
}

.app-subtitle {
    font-size: 17px;
    opacity: 0.75;
}


/* ----------------------------------------------------------
   SECTION HEADERS
---------------------------------------------------------- */

.section-title {
    font-size: 22px;
    font-weight: 700;
    margin-bottom: 5px;
}

.section-description {
    opacity: 0.7;
    margin-bottom: 15px;
}


/* ----------------------------------------------------------
   SYMPTOM BOX
---------------------------------------------------------- */

.symptom-box {
    border-radius: 16px !important;
    padding: 10px !important;
}


/* ----------------------------------------------------------
   BUTTONS
---------------------------------------------------------- */

.predict-button {
    min-height: 55px !important;
    border-radius: 12px !important;
    font-size: 17px !important;
    font-weight: 700 !important;
}

.clear-button {
    min-height: 55px !important;
    border-radius: 12px !important;
}


/* ----------------------------------------------------------
   RESULT CARD
---------------------------------------------------------- */

.result-card {
    border-radius: 18px;
    padding: 25px;
    margin-top: 10px;
    border: 1px solid rgba(100, 100, 100, 0.2);
    background: rgba(120, 120, 120, 0.05);
}

.result-header {
    display: flex;
    align-items: center;
    gap: 18px;
}

.result-icon {
    font-size: 48px;
}

.result-label {
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 1.5px;
    opacity: 0.65;
}

.result-disease {
    font-size: 30px;
    font-weight: 800;
    margin-top: 4px;
}


/* ----------------------------------------------------------
   CONFIDENCE
---------------------------------------------------------- */

.confidence-section {
    margin-top: 25px;
}

.confidence-top {
    display: flex;
    justify-content: space-between;
    font-size: 16px;
    margin-bottom: 9px;
}

.confidence-top strong {
    font-size: 20px;
}

.confidence-bar {
    width: 100%;
    height: 12px;
    border-radius: 20px;
    background: rgba(128, 128, 128, 0.2);
    overflow: hidden;
}

.confidence-fill {
    height: 100%;
    border-radius: 20px;
    background: linear-gradient(
        90deg,
        #2563eb,
        #16a34a
    );
}

.confidence-bottom {
    margin-top: 9px;
    font-size: 14px;
    opacity: 0.7;
}


/* ----------------------------------------------------------
   EMPTY RESULT
---------------------------------------------------------- */

.empty-result {
    text-align: center;
    padding: 55px 20px;
    border-radius: 18px;
    border: 1px dashed rgba(128, 128, 128, 0.4);
}

.empty-icon {
    font-size: 45px;
    margin-bottom: 10px;
}

.empty-result h2 {
    margin-bottom: 5px;
}

.empty-result p {
    opacity: 0.7;
}


/* ----------------------------------------------------------
   FOOTER
---------------------------------------------------------- */

.disclaimer {
    text-align: center;
    font-size: 13px;
    opacity: 0.65;
    padding: 20px;
    line-height: 1.6;
}


/* ----------------------------------------------------------
   RESPONSIVE
---------------------------------------------------------- */

@media (max-width: 700px) {

    .app-title {
        font-size: 28px;
    }

    .app-subtitle {
        font-size: 14px;
    }

    .result-disease {
        font-size: 23px;
    }

}

"""


# ============================================================
# GRADIO APPLICATION
# ============================================================

with gr.Blocks(
    css=custom_css,
    title="Smart Disease Prediction System"
) as demo:


    # ========================================================
    # HEADER
    # ========================================================

    gr.HTML(
        """
        <div class="app-header">

            <div class="app-title">
                🩺 Smart Disease Prediction System
            </div>

            <div class="app-subtitle">
                AI-assisted symptom-based disease prediction
                using Machine Learning
            </div>

        </div>
        """
    )


    # ========================================================
    # MAIN INPUT AREA
    # ========================================================

    with gr.Row():

        with gr.Column(
            scale=1,
            variant="panel"
        ):

            gr.Markdown(
                """
                ### 🔍 Select Your Symptoms

                Choose the symptoms currently experienced.
                You can search through the available symptoms.
                """,
                elem_classes="section-description"
            )


            symptom_input = gr.Dropdown(
                choices=symptom_choices,
                multiselect=True,
                filterable=True,
                label="Symptoms",
                info="Search and select one or more symptoms",
                elem_classes="symptom-box"
            )


            with gr.Row():

                predict_button = gr.Button(
                    "🔮 Predict Disease",
                    variant="primary",
                    elem_classes="predict-button"
                )

                clear_button = gr.Button(
                    "🧹 Clear",
                    variant="secondary",
                    elem_classes="clear-button"
                )


    # ========================================================
    # PREDICTION RESULT
    # ========================================================

    gr.Markdown("## 🎯 Prediction Result")


    result_html = gr.HTML(
        """
        <div class="empty-result">
            <div class="empty-icon">🩺</div>
            <h2>Ready for prediction</h2>
            <p>Select your symptoms and click <b>Predict Disease</b>.</p>
        </div>
        """
    )


    # Hidden/simple outputs for accessibility and easy display
    with gr.Row():

        predicted_output = gr.Textbox(
            label="Predicted Disease",
            value="—",
            interactive=False
        )

        confidence_output = gr.Textbox(
            label="Confidence",
            value="0.00%",
            interactive=False
        )


    # ========================================================
    # TOP 3 PREDICTIONS
    # ========================================================

    gr.Markdown("## 🏆 Top 3 Predictions")


    top3_output = gr.Dataframe(
        headers=[
            "Rank",
            "Disease",
            "Confidence"
        ],
        datatype=[
            "number",
            "str",
            "str"
        ],
        value=[
            ["1", "—", "0.00%"],
            ["2", "—", "0.00%"],
            ["3", "—", "0.00%"]
        ],
        interactive=False,
        wrap=True
    )


    # ========================================================
    # MODEL COMPARISON
    # ========================================================

    gr.Markdown("## 🤖 Model Comparison")

    gr.Markdown(
        """
        The system compares predictions from the three
        individual classifiers and the final Soft Voting Ensemble.
        """
    )


    comparison_output = gr.Dataframe(
        headers=[
            "Model",
            "Prediction",
            "Status",
            "Confidence"
        ],
        datatype=[
            "str",
            "str",
            "str",
            "str"
        ],
        value=[
            [
                "Logistic Regression",
                "—",
                "—",
                "—"
            ],
            [
                "Random Forest",
                "—",
                "—",
                "—"
            ],
            [
                "SVM",
                "—",
                "—",
                "—"
            ],
            [
                "Soft Voting Ensemble",
                "—",
                "—",
                "—"
            ]
        ],
        interactive=False,
        wrap=True
    )


    # ========================================================
    # METHODOLOGY
    # ========================================================

    with gr.Accordion(
        "ℹ️ About the System",
        open=False
    ):

        gr.Markdown(
            """
            ### Machine Learning Pipeline

            **Input Symptoms**

            ↓

            **130 Symptom Features**

            ↓

            **Machine Learning Models**

            - Logistic Regression
            - Random Forest
            - Support Vector Machine

            ↓

            **Soft Voting Ensemble**

            ↓

            **Disease Prediction**

            ### Explainability

            Random Forest feature importance is used during
            model analysis to identify symptom features that
            the model relies on most.

            ### Important

            Model confidence represents the model's prediction
            score for this dataset. It is **not a clinical
            probability or medical certainty**.
            """
        )


    # ========================================================
    # DISCLAIMER
    # ========================================================

    gr.HTML(
        """
        <div class="disclaimer">

            ⚠️ <b>Academic Prototype Disclaimer:</b><br>

            This system is developed for educational and
            academic purposes. It is not a substitute for
            professional medical diagnosis, treatment, or
            medical advice. Please consult a qualified
            healthcare professional for medical decisions.

        </div>
        """
    )


    # ========================================================
    # BUTTON EVENTS
    # ========================================================

    predict_button.click(
        fn=predict_disease,
        inputs=symptom_input,
        outputs=[
            result_html,
            predicted_output,
            confidence_output,
            top3_output,
            comparison_output
        ]
    )


    clear_button.click(
        fn=clear_results,
        inputs=[],
        outputs=[
            symptom_input,
            result_html,
            predicted_output,
            confidence_output,
            top3_output,
            comparison_output
        ]
    )


# ============================================================
# LAUNCH APPLICATION
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("SMART DISEASE PREDICTION SYSTEM")
    print("=" * 60)
    print("Starting Gradio application...")
    print("=" * 60)

    demo.launch()