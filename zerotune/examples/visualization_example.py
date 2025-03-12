"""
Visualization example for ZeroTune.

This example demonstrates:
1. How to use matplotlib to visualize hyperparameter optimization results
2. Different plot types for comparing model performance
3. Visualization of dataset characteristics impact on performance
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.datasets import load_breast_cancer, load_wine, load_digits
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

# Try to import matplotlib, but handle missing dependencies gracefully
try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    VISUALIZATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Visualization will not be available: {str(e)}")
    print("To fix, try: poetry add 'importlib_resources<6.0.0' 'matplotlib<3.10'")
    VISUALIZATION_AVAILABLE = False
    print("This example requires matplotlib to work correctly. Please install it and try again.")
    import sys
    sys.exit(1)

# Import ZeroTune - two approaches provided:
# Approach 1: Use proper package import if installed
try:
    from zerotune import (
        ZeroTunePredictor, 
        calculate_dataset_meta_parameters,
        get_available_models
    )
    print("Using installed ZeroTune package")
# Approach 2: Use relative import for development
except ImportError:
    import sys
    # Add the parent directory of the current file to the path
    module_path = str(Path(__file__).resolve().parent.parent.parent)
    if module_path not in sys.path:
        sys.path.insert(0, module_path)
    from zerotune import (
        ZeroTunePredictor, 
        calculate_dataset_meta_parameters,
        get_available_models
    )
    print("Using development ZeroTune import")

# Create output directory for visualizations
os.makedirs("visualization_output", exist_ok=True)

# Load different datasets for comparison
print("Loading datasets...")
datasets = {
    "breast_cancer": load_breast_cancer(),
    "wine": load_wine(),
    "digits": load_digits()
}

# Process each dataset
dataset_results = {}
dataset_meta_features = {}

for name, data in datasets.items():
    print(f"\nProcessing {name} dataset...")
    X = pd.DataFrame(data.data, columns=data.feature_names if hasattr(data, 'feature_names') else None)
    y = pd.Series(data.target)
    
    # Calculate meta-features
    meta_features = calculate_dataset_meta_parameters(X, y)
    dataset_meta_features[name] = meta_features
    print(f"  Shape: {X.shape}")
    print(f"  Classes: {len(np.unique(y))}")
    
    # Create ZeroTune predictor
    try:
        predictor = ZeroTunePredictor(model_name="decision_tree")
        
        # Predict hyperparameters
        hyperparams = predictor.predict(X, y)
        print("  Predicted hyperparameters:")
        for param, value in hyperparams.items():
            print(f"    - {param}: {value}")
        
        # Evaluate with ZeroTune hyperparameters
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        model = DecisionTreeClassifier(**hyperparams, random_state=42)
        scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv)
        
        # Evaluate with default hyperparameters
        default_model = DecisionTreeClassifier(random_state=42)
        default_scores = cross_val_score(default_model, X, y, scoring='roc_auc', cv=cv)
        
        # Calculate improvement
        improvement = (np.mean(scores) - np.mean(default_scores)) / np.mean(default_scores) * 100
        
        # Store results
        dataset_results[name] = {
            "zerotune_mean": np.mean(scores),
            "zerotune_std": np.std(scores),
            "default_mean": np.mean(default_scores),
            "default_std": np.std(default_scores),
            "improvement": improvement,
            "hyperparams": hyperparams
        }
        
        print(f"  ROC AUC with ZeroTune: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
        print(f"  ROC AUC with Default: {np.mean(default_scores):.4f} ± {np.std(default_scores):.4f}")
        print(f"  Improvement: {improvement:.2f}%")
        
    except Exception as e:
        print(f"  Error processing {name} dataset: {e}")

# Create a DataFrame for visualization
results_df = pd.DataFrame({
    'Dataset': list(dataset_results.keys()),
    'ZeroTune Mean': [r['zerotune_mean'] for r in dataset_results.values()],
    'ZeroTune Std': [r['zerotune_std'] for r in dataset_results.values()],
    'Default Mean': [r['default_mean'] for r in dataset_results.values()],
    'Default Std': [r['default_std'] for r in dataset_results.values()],
    'Improvement (%)': [r['improvement'] for r in dataset_results.values()]
})

# Save results to CSV
results_df.to_csv("visualization_output/dataset_comparison.csv", index=False)
print("\nResults saved to visualization_output/dataset_comparison.csv")

# Create visualizations
print("\nCreating visualizations...")

# 1. Bar chart comparison across datasets
plt.figure(figsize=(12, 8))
datasets_names = list(dataset_results.keys())
x = np.arange(len(datasets_names))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
rects1 = ax.bar(x - width/2, 
                [r['default_mean'] for r in dataset_results.values()], 
                width, 
                yerr=[r['default_std'] for r in dataset_results.values()],
                label='Default', 
                color='lightgray', 
                capsize=10)
rects2 = ax.bar(x + width/2, 
                [r['zerotune_mean'] for r in dataset_results.values()], 
                width, 
                yerr=[r['zerotune_std'] for r in dataset_results.values()],
                label='ZeroTune', 
                color='green', 
                capsize=10)

# Add text labels
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

autolabel(rects1)
autolabel(rects2)

# Add improvement annotations
for i, name in enumerate(dataset_results.keys()):
    improvement = dataset_results[name]['improvement']
    plt.annotate(f'+{improvement:.1f}%', 
                xy=(i + width/2, dataset_results[name]['zerotune_mean']), 
                xytext=(0, 10),
                textcoords="offset points",
                ha='center', 
                fontsize=9, 
                fontweight='bold',
                color='darkgreen')

ax.set_ylabel('ROC AUC Score')
ax.set_title('Performance Comparison Across Datasets')
ax.set_xticks(x)
ax.set_xticklabels(datasets_names)
ax.legend()
ax.set_ylim(0.9, 1.0)  # Adjust as needed

plt.tight_layout()
plt.savefig('visualization_output/dataset_comparison_bar.png')
print("Bar chart saved as 'visualization_output/dataset_comparison_bar.png'")

# 2. Scatter plot showing relationship between number of features and improvement
plt.figure(figsize=(10, 6))
x_values = [dataset_meta_features[name]['n_features'] for name in dataset_results.keys()]
y_values = [dataset_results[name]['improvement'] for name in dataset_results.keys()]
sizes = [dataset_meta_features[name]['n_samples']/10 for name in dataset_results.keys()]

plt.scatter(x_values, y_values, s=sizes, alpha=0.7, c='green')

for i, name in enumerate(dataset_results.keys()):
    plt.annotate(name, 
                (x_values[i], y_values[i]),
                xytext=(5, 5),
                textcoords='offset points')

plt.xlabel('Number of Features')
plt.ylabel('Improvement (%)')
plt.title('Relationship Between Number of Features and Improvement')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('visualization_output/features_vs_improvement.png')
print("Scatter plot saved as 'visualization_output/features_vs_improvement.png'")

# 3. Hyperparameter comparison across datasets
# Create a multi-panel figure
fig = plt.figure(figsize=(12, 10))
gs = gridspec.GridSpec(2, 2)

# Define the hyperparameters to compare
hyperparams_to_compare = ['max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features']

for i, param in enumerate(hyperparams_to_compare):
    ax = fig.add_subplot(gs[i//2, i%2])
    values = [dataset_results[name]['hyperparams'][param] for name in dataset_results.keys()]
    
    # Create the bar chart
    ax.bar(datasets_names, values, color='teal', alpha=0.7)
    
    # Add value labels
    for j, v in enumerate(values):
        ax.text(j, v, f'{v:.3f}' if isinstance(v, float) else str(v), 
                ha='center', va='bottom', fontsize=9)
    
    ax.set_title(f'{param} Comparison')
    ax.set_ylabel('Value')
    if i >= 2:  # Only add x labels for bottom plots
        ax.set_xticklabels(datasets_names, rotation=45, ha='right')
    else:
        ax.set_xticklabels([])

plt.tight_layout()
plt.savefig('visualization_output/hyperparameter_comparison.png')
print("Hyperparameter comparison plot saved as 'visualization_output/hyperparameter_comparison.png'")

# Show all plots (if in interactive environment)
plt.show()

print("\nVisualization completed! All plots saved to the 'visualization_output' directory.")
print("\nFor more examples, see the ZeroTune documentation:"
      "https://github.com/yourusername/zerotune") 