import os
import joblib
import pandas as pd
import gradio as gr

# Load trained bundle
bundle_path = os.path.join(os.path.dirname(__file__), "student_dropout_model_bundle.joblib")
bundle = joblib.load(bundle_path)
model = bundle["model"]
feature_names = bundle["feature_names"]
LABEL_MAPPING = bundle["label_mapping"]
X_test = bundle["X_test"]

def predict_student_dropout(
    age,
    curricular_approved,
    curricular_enrolled,
    curricular_grade,
    tuition_fees,
    scholarship,
    admission_grade
):
    # Baseline test row
    input_data = X_test.iloc[0].to_dict()

    # Update with inputs from interface
    input_data["Age at enrollment"] = age
    input_data["Curricular units 1st sem (approved)"] = curricular_approved
    input_data["Curricular units 1st sem (enrolled)"] = curricular_enrolled
    input_data["Curricular units 1st sem (grade)"] = curricular_grade
    input_data["Tuition fees up to date"] = int(tuition_fees)
    input_data["Scholarship holder"] = int(scholarship)
    input_data["Admission grade"] = admission_grade

    input_df = pd.DataFrame([input_data])[feature_names]

    pred_idx = model.predict(input_df)[0]
    pred_probs = model.predict_proba(input_df)[0]

    predicted_label = LABEL_MAPPING[pred_idx]

    confidence_dict = {
        LABEL_MAPPING[i]: float(pred_probs[i])
        for i in range(len(LABEL_MAPPING))
    }

    return predicted_label, confidence_dict

custom_css = """
body { background-color: #0f172a; color: #f8fafc; }
.gradio-container {
    max-width: 900px !important;
    border-radius: 16px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    padding: 28px !important;
    background: #1e293b;
    border: 1px solid #334155;
}
.gr-button-primary {
    background: linear-gradient(135deg, #4f46e5 0%, #3b82f6 100%) !important;
    border: none !important;
    font-weight: 700 !important;
    color: white !important;
    border-radius: 8px !important;
}
"""

demo = gr.Interface(
    fn=predict_student_dropout,
    inputs=[
        gr.Slider(minimum=17, maximum=60, value=20, step=1, label="Age at enrollment"),
        gr.Slider(minimum=0, maximum=10, value=4, step=1, label="Curricular units 1st sem (approved)"),
        gr.Slider(minimum=0, maximum=15, value=6, step=1, label="Curricular units 1st sem (enrolled)"),
        gr.Slider(minimum=0.0, maximum=20.0, value=11.5, step=0.1, label="Curricular units 1st sem (grade)"),
        gr.Radio(choices=[("Yes", 1), ("No", 0)], value=1, label="Tuition fees up to date"),
        gr.Radio(choices=[("Yes", 1), ("No", 0)], value=0, label="Scholarship holder"),
        gr.Slider(minimum=0.0, maximum=200.0, value=120.0, step=0.5, label="Admission grade")
    ],
    outputs=[
        gr.Textbox(label="Predicted Academic Status"),
        gr.Label(label="Class Probabilities")
    ],
    title="🎓 Student Dropout Risk Predictor",
    description="Early-warning machine learning system for predicting higher education student outcomes (Dropout, Enrolled, Graduate) based on first-semester academic and demographic indicators.",
    css=custom_css,
    theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="blue")
)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
