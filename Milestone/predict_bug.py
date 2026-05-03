# predict_bug.py
# Interactive script to predict UI/UX Bug Severity AND Priority

import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

def main():
    print("\n" + "=" * 60)
    print("AI Frontend UI/UX Bug Predictor (Severity & Priority)")
    print("=" * 60)

    try:
        model_severity = joblib.load("severity_rf_model.joblib")
        model_priority = joblib.load("priority_rf_model.joblib")
        encoders = joblib.load("data_encoders.joblib")
        
        feature_encoders = encoders['features']
        target_encoder_sev = encoders['target_severity']
        target_encoder_pri = encoders['target_priority']
    except FileNotFoundError:
        print("Error: Could not find model files. Please run model_training.py first!")
        return

    print("\nPlease enter the details of the new bug:")
    
    app_name = input("App Name (e.g., Taskmanager, Shopease): ").strip().capitalize()
    module = input("Module (e.g., Checkout, Sidebar, Navbar): ").strip().capitalize()
    bug_type = input("Bug Type (e.g., Crash, Color Mismatch, Layout): ").strip().title()
    device_type = input("Device (Desktop, Mobile, Tablet): ").strip().capitalize()
    browser = input("Browser (Chrome, Safari, Firefox, Edge): ").strip().capitalize()
    os_name = input("OS (Windows, Macos, Android, Ios): ").strip().capitalize()
    reported_by = input("Reported By (User, Qa, Developer): ").strip().capitalize()
    screenshot = input("Screenshot Available? (True/False): ").strip().capitalize()
    
    try:
        time_to_detect = int(input("Time to detect (seconds, e.g., 45): "))
        time_to_fix = int(input("Estimated time to fix (seconds, e.g., 500): "))
        user_impact = int(input("User impact score (1-10): "))
    except ValueError:
        print("Invalid number entered. Defaulting numerical values to averages.")
        time_to_detect, time_to_fix, user_impact = 30, 400, 5

    resolved = "False"

    input_df = pd.DataFrame([{
        'app_name': app_name, 'module': module, 'bug_type': bug_type,
        'device_type': device_type, 'browser': browser, 'os': os_name,
        'screenshot_available': screenshot, 'time_to_detect_sec': time_to_detect,
        'time_to_fix_sec': time_to_fix, 'user_impact_score': user_impact,
        'reported_by': reported_by, 'resolved': resolved
    }])

    for col in feature_encoders.keys():
        le = feature_encoders[col]
        try:
            input_df[col] = le.transform(input_df[col].astype(str))
        except ValueError:
            input_df[col] = le.transform([le.classes_[0]])

    # Make Both Predictions
    pred_sev_encoded = model_severity.predict(input_df)
    pred_pri_encoded = model_priority.predict(input_df)
    
    severity_text = target_encoder_sev.inverse_transform(pred_sev_encoded)[0]
    priority_text = target_encoder_pri.inverse_transform(pred_pri_encoded)[0]

    print("\n" + "-" * 50)
    print("AI PREDICTION RESULTS")
    print("-" * 50)
    print(f"Predicted Severity: {severity_text.upper()}")
    print(f"Predicted Priority: {priority_text.upper()}")
    print("-" * 50 + "\n")

if __name__ == "__main__":
    main()