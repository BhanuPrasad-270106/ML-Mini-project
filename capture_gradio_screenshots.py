import os
import time
import threading
import joblib
import pandas as pd
import gradio as gr
from playwright.sync_api import sync_playwright

# 1. Load trained model bundle
bundle = joblib.load("student_dropout_model_bundle.joblib")
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
    input_data = X_test.iloc[0].to_dict()

    input_data["Age at enrollment"] = float(age)
    input_data["Curricular units 1st sem (approved)"] = float(curricular_approved)
    input_data["Curricular units 1st sem (enrolled)"] = float(curricular_enrolled)
    input_data["Curricular units 1st sem (grade)"] = float(curricular_grade)
    input_data["Tuition fees up to date"] = int(tuition_fees)
    input_data["Scholarship holder"] = int(scholarship)
    input_data["Admission grade"] = float(admission_grade)

    input_df = pd.DataFrame([input_data])[feature_names]

    pred_idx = model.predict(input_df)[0]
    pred_probs = model.predict_proba(input_df)[0]

    predicted_label = LABEL_MAPPING[pred_idx]

    confidence_dict = {
        LABEL_MAPPING[i]: float(pred_probs[i])
        for i in range(len(LABEL_MAPPING))
    }

    return predicted_label, confidence_dict

# Professional styling matching notebook Block 29
custom_css = """
body { 
    background-color: #f1f5f9 !important; 
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif !important;
}
.gradio-container {
    max-width: 880px !important;
    border-radius: 16px !important;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1) !important;
    padding: 28px !important;
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    margin-top: 20px !important;
    margin-bottom: 20px !important;
}
h1 {
    color: #0f172a !important;
    font-size: 26px !important;
    font-weight: 800 !important;
    margin-bottom: 8px !important;
}
p {
    color: #475569 !important;
    font-size: 14px !important;
}
.gr-button-primary {
    background: linear-gradient(135deg, #4f46e5 0%, #3b82f6 100%) !important;
    border: none !important;
    font-weight: 700 !important;
    color: #ffffff !important;
    padding: 10px 20px !important;
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
    css=custom_css
)

def start_gradio():
    demo.launch(server_name="127.0.0.1", server_port=7875, share=False, prevent_thread_lock=True)

# Start Gradio in background thread
t = threading.Thread(target=start_gradio, daemon=True)
t.start()
time.sleep(3)

os.makedirs("images", exist_ok=True)

with sync_playwright() as p:
    browser = p.chromium.launch(channel="msedge", headless=True)
    page = browser.new_page(viewport={"width": 1280, "height": 950})
    page.goto("http://127.0.0.1:7875")
    page.wait_for_selector("button:has-text('Submit')")
    time.sleep(2)

    scenarios = [
        {
            "name": "gradio_dropout_prediction.png",
            "age": "25",
            "approved": "0",
            "enrolled": "6",
            "grade": "0",
            "tuition_yes": False,
            "scholarship_yes": False,
            "admission": "115.0",
            "expected": "Dropout"
        },
        {
            "name": "gradio_enrolled_prediction.png",
            "age": "18",
            "approved": "5",
            "enrolled": "6",
            "grade": "10",
            "tuition_yes": True,
            "scholarship_yes": False,
            "admission": "120.7",
            "expected": "Enrolled"
        },
        {
            "name": "gradio_graduate_prediction.png",
            "age": "18",
            "approved": "6",
            "enrolled": "6",
            "grade": "15.5",
            "tuition_yes": True,
            "scholarship_yes": True,
            "admission": "150.0",
            "expected": "Graduate"
        }
    ]

    for sc in scenarios:
        print(f"\n--- Setting inputs for {sc['expected']} ---")

        # Slider number inputs
        number_inputs = page.query_selector_all("input[type='number']")
        if len(number_inputs) >= 5:
            # 0: Age, 1: Approved, 2: Enrolled, 3: Grade, 4: Admission
            number_inputs[0].fill(sc["age"])
            number_inputs[1].fill(sc["approved"])
            number_inputs[2].fill(sc["enrolled"])
            number_inputs[3].fill(sc["grade"])
            number_inputs[4].fill(sc["admission"])

        # Radio button labels
        yes_labels = page.query_selector_all("label:has-text('Yes')")
        no_labels = page.query_selector_all("label:has-text('No')")

        if len(yes_labels) >= 2 and len(no_labels) >= 2:
            if sc["tuition_yes"]:
                yes_labels[0].click()
            else:
                no_labels[0].click()

            if sc["scholarship_yes"]:
                yes_labels[1].click()
            else:
                no_labels[1].click()

        time.sleep(0.5)
        # Click submit
        page.click("button:has-text('Submit')")
        
        # Wait until processing is finished and the output is updated
        print(f"Waiting for prediction '{sc['expected']}' to appear in output...")
        for _ in range(30):
            time.sleep(0.5)
            # Check textbox value or text
            text_boxes = page.query_selector_all("textarea, input[readonly]")
            matched = False
            for tb in text_boxes:
                val = tb.input_value() if hasattr(tb, "input_value") else ""
                if sc["expected"].lower() in val.lower():
                    matched = True
                    break
            
            # Also check if loading indicator is gone
            loadings = page.query_selector_all(".loading, .progress")
            if matched and len(loadings) == 0:
                print(f"Output confirmed: {sc['expected']}")
                time.sleep(1.0)
                break

        target_path = os.path.join("images", sc["name"])
        container = page.query_selector(".gradio-container")
        if container:
            container.screenshot(path=target_path)
        else:
            page.screenshot(path=target_path)
            
        print(f"Captured: {target_path}")

    browser.close()
    print("\nAll Gradio screenshots captured successfully!")
