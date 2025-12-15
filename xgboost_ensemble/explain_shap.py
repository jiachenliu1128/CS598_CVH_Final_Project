import xgboost as xgb
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from train_ensemble import load_and_prep_data

def generate_shap_plots():
    # 1. Load Data
    train_df, val_df, test_df = load_and_prep_data()
    
    # 2. Load Model
    model = xgb.Booster()
    model.load_model("models/xgboost_ensemble.json")
    
    # 3. Prepare Test Data
    features = ['image_score', 'age', 'density', 'implant', 'site_id']
    X_test = test_df[features]
    
    # 4. Compute SHAP Values
    print("Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # 5. Generate Summary Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title("SHAP Summary Plot (Global Importance)")
    plt.tight_layout()
    plt.savefig("../results/figures/shap_summary.png")
    print("Saved ../results/figures/shap_summary.png")
    
    # 6. Generate Dependence Plots
    # Show how age affects prediction
    plt.figure(figsize=(8, 5))
    shap.dependence_plot("age", shap_values, X_test, interaction_index=None, show=False)
    plt.title("SHAP Dependence: Age")
    plt.tight_layout()
    plt.savefig("../results/figures/shap_dependence_age.png")
    print("Saved ../results/figures/shap_dependence_age.png")
    
    # Show how image_score affects prediction (should be linear/monotonic)
    plt.figure(figsize=(8, 5))
    shap.dependence_plot("image_score", shap_values, X_test, interaction_index=None, show=False)
    plt.title("SHAP Dependence: Image Score")
    plt.tight_layout()
    plt.savefig("../results/figures/shap_dependence_image.png")
    print("Saved ../results/figures/shap_dependence_image.png")
    
    # 7. Individual Explanation (Example)
    # Find a positive case
    pos_idx = test_df[test_df['cancer'] == 1].index[0]
    # We need iloc relative to X_test (which is already a filtered DF)
    # Since X_test indices match test_df indices, we can look up by index label
    # but SHAP expects integer location for the numpy array.
    
    # Reset index to make iloc easy
    X_test_reset = X_test.reset_index(drop=True)
    y_test_reset = test_df['cancer'].reset_index(drop=True)
    
    # Find first cancer case
    pos_loc = y_test_reset[y_test_reset == 1].index[0]
    
    print(f"\nExplaining positive case at index {pos_loc}")
    print(X_test_reset.iloc[pos_loc])
    
    # Force Plot
    shap.force_plot(
        explainer.expected_value, 
        shap_values[pos_loc], 
        X_test_reset.iloc[pos_loc],
        matplotlib=True,
        show=False
    )
    plt.savefig("../results/figures/shap_force_positive.png")
    print("Saved ../results/figures/shap_force_positive.png")

if __name__ == "__main__":
    generate_shap_plots()

