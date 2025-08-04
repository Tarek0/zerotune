#!/usr/bin/env python3
"""
XGBoost Knowledge Base Builder & Zero-Shot Predictor Training/Evaluation

CURRENT STATUS (completed):
✅ Knowledge base building from 2 test datasets (31: credit-g, 38: sick)
✅ Zero-shot predictor training from knowledge base 
✅ Evaluation on 10 unseen datasets with excellent results (avg AUC: 0.8754)
✅ Proper data leakage prevention (training vs evaluation datasets are separate)
✅ Consistent naming conventions for KB and models

ARCHITECTURE:
- zerotune/core/predictor_training.py: Training functionality  
- zerotune/predictors.py: Predictor class for inference
- models/xgboost_binary_classifier_xgb_kb_v1_test.joblib: Trained model
- knowledge_base/kb_xgboost_xgb_kb_v1_test.json: Training data

CURRENT PERFORMANCE:
- Test predictor (2 training datasets): Average AUC 0.8754 ± 0.1304 on 10 unseen datasets
- Detailed results on unseen datasets:
  * 917 (fri_c1_1000_25): 0.9790
  * 1049 (pc4): 0.9571  
  * 1111 (KDDCup09_appetency): 0.8447
  * 1464 (blood-transfusion): 0.6809
  * 1494 (qsar-biodeg): 0.9338
  * 1510 (wdbc): 0.9918 ← Best
  * 1558 (bank-marketing): 0.9179
  * 4534 (PhishingWebsites): 0.9913
  * 23381 (dresses-sales): 0.5899 ← Worst
  * 40536 (SpeedDating): 0.8675
- 100% success rate, 8/10 datasets achieved AUC > 0.8

NEXT STEPS TO IMPROVE:
1. Build full knowledge base: python xgb_experiment.py full      # 15 datasets, 20 iterations each
2. Train full predictor:      python xgb_experiment.py train-full
3. Evaluate full predictor:   python xgb_experiment.py eval-full
4. Compare test vs full predictor performance

DATASET COLLECTIONS:
- Training (test): [31, 38] - 2 datasets for quick development
- Training (full): [31, 38, 44, 52, 151, 179, 298, 846, 1053, 1112, 1120, 1128, 1220, 40900, 45038] - 15 datasets
- Evaluation (unseen): [917, 1049, 1111, 1464, 1494, 1510, 1558, 4534, 23381, 40536] - 10 datasets

This script builds a comprehensive knowledge base for training zero-shot hyperparameter predictors.
It runs hyperparameter optimization on multiple datasets and stores:
- Dataset meta-features (n_samples, n_features, imbalance_ratio, etc.)
- Optimal hyperparameters found through HPO
- Performance scores achieved

The resulting knowledge base can be used to train zero-shot predictors.
"""

from zerotune import ZeroTune
from zerotune.core.predictor_training import train_predictor_from_knowledge_base
from zerotune.core.data_loading import fetch_open_ml_data, prepare_data
from zerotune.core.feature_extraction import calculate_dataset_meta_parameters
from zerotune.core.utils import convert_to_dataframe
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
import joblib
import numpy as np
import sys
import os
from datetime import datetime

# Experiment configuration
EXPERIMENT_ID = "xgb_kb_v1"

# Dataset collection for knowledge base building
FULL_DATASET_COLLECTION = [
    31,    # credit-g (1,000 × 20, imbalance: 2.333)
    38,    # sick (3,772 × 29, imbalance: 15.329)
    44,    # spambase (4,601 × 57, imbalance: 1.538)
    52,    # trains (10 × 32, imbalance: 1.000)
    151,   # electricity (45,312 × 8, imbalance: 1.355)
    179,   # adult (48,842 × 14, imbalance: 3.179)
    298,   # coil2000 (9,822 × 85, imbalance: 15.761)
    846,   # elevators (16,599 × 18, imbalance: 2.236)
    1053,  # jm1 (10,885 × 21, imbalance: 4.169)
    1112,  # KDDCup09_churn (50,000 × 230, imbalance: 12.617)
    1120,  # MagicTelescope (19,020 × 10, imbalance: 1.844)
    1128,  # OVA_Breast (1,545 × 10,935, imbalance: 3.491)
    1220,  # Click_prediction_small (39,948 × 9, imbalance: 4.938)
    40900, # Satellite (5,100 × 36, imbalance: 67.000)
    45038  # road-safety (111,762 × 32, imbalance: 1.000)
]

# Test subset for development (2 datasets)
TEST_DATASET_COLLECTION = [31, 38]  # credit-g, sick

# Unseen datasets for testing trained predictors (avoid data leakage)
UNSEEN_TEST_DATASETS = [917, 1049, 1111, 1464, 1494, 1510, 1558, 4534, 23381, 40536]

def build_knowledge_base(dataset_ids, n_iter=20, mode="test"):
    """
    Build knowledge base by running HPO on specified datasets.
    
    Args:
        dataset_ids: List of OpenML dataset IDs
        n_iter: Number of HPO iterations per dataset
        mode: "test" or "full" for different configurations
    """
    print("ZEROTUNE KNOWLEDGE BASE BUILDER")
    print("=" * 60)
    print(f"Experiment ID: {EXPERIMENT_ID}")
    print(f"Mode: {mode}")
    print(f"Building knowledge base with {len(dataset_ids)} datasets")
    print(f"HPO iterations per dataset: {n_iter}")
    print(f"Dataset IDs: {dataset_ids}")
    
    # Create knowledge base directory
    os.makedirs("knowledge_base", exist_ok=True)
    kb_path = f"knowledge_base/kb_xgboost_{EXPERIMENT_ID}_{mode}.json"
    print(f"Knowledge base will be saved to: {kb_path}")
    
    # Initialize ZeroTune for knowledge base building
    print(f"\nInitializing ZeroTune for XGBoost knowledge base building...")
    zt = ZeroTune(model_type='xgboost', kb_path=kb_path)
    
    # Build knowledge base
    print(f"\nStarting knowledge base building...")
    print("This will run HPO on each dataset to collect optimal hyperparameters")
    print("-" * 60)
    
    try:
        kb = zt.build_knowledge_base(
            dataset_ids=dataset_ids, 
            n_iter=n_iter,
            verbose=True
        )
        
        print(f"\n{'='*60}")
        print("KNOWLEDGE BASE BUILDING COMPLETED")
        print(f"{'='*60}")
        print(f"✅ Successfully processed {len(dataset_ids)} datasets")
        print(f"📁 Knowledge base saved to: {kb_path}")
        
        # Print knowledge base summary
        if kb and 'meta_features' in kb:
            print(f"📊 Knowledge base contains {len(kb['meta_features'])} dataset entries")
        if kb and 'results' in kb:
            print(f"🎯 Knowledge base contains {len(kb['results'])} optimization results")
            
        return kb_path, kb
        
    except Exception as e:
        print(f"\n❌ Error during knowledge base building: {str(e)}")
        return None, None

def train_zero_shot_predictor(kb_path=None, mode="test"):
    """
    Train a zero-shot predictor from a knowledge base.
    
    Args:
        kb_path: Path to knowledge base (auto-generated if None)
        mode: "test" or "full" to match the KB naming convention
    """
    print("ZEROTUNE ZERO-SHOT PREDICTOR TRAINER")
    print("=" * 60)
    print(f"Experiment ID: {EXPERIMENT_ID}")
    print(f"Mode: {mode}")
    
    # Auto-generate KB path if not provided
    if kb_path is None:
        kb_path = f"knowledge_base/kb_xgboost_{EXPERIMENT_ID}_{mode}.json"
    
    print(f"Using knowledge base: {kb_path}")
    
    if not os.path.exists(kb_path):
        print(f"❌ Knowledge base not found: {kb_path}")
        print("Please build the knowledge base first using:")
        print(f"  python xgb_experiment.py {mode}")
        return None
    
    try:
        # Train the predictor
        print(f"\nTraining zero-shot predictor...")
        print("This will create a model that can predict optimal hyperparameters for new datasets")
        print("-" * 60)
        
        model_path = train_predictor_from_knowledge_base(
            kb_path=kb_path,
            model_name="xgboost",
            task_type="binary",
            output_dir="models",
            exp_id=f"{EXPERIMENT_ID}_{mode}"
        )
        
        print(f"\n{'='*60}")
        print("ZERO-SHOT PREDICTOR TRAINING COMPLETED")
        print(f"{'='*60}")
        print("✅ Successfully trained zero-shot predictor")
        print(f"📁 Model saved to: {model_path}")
        print(f"🎯 Ready for zero-shot hyperparameter prediction")
        
        return model_path
        
    except Exception as e:
        print(f"\n❌ Error during predictor training: {str(e)}")
        return None

def test_zero_shot_predictor(model_path=None, mode="test", test_dataset_ids=None):
    """
    Test a trained zero-shot predictor on unseen datasets.
    
    Args:
        model_path: Path to trained model (auto-generated if None)
        mode: "test" or "full" to match the model naming convention
        test_dataset_ids: List of dataset IDs to test on (uses UNSEEN_TEST_DATASETS if None)
    """
    print("ZEROTUNE ZERO-SHOT PREDICTOR EVALUATION")
    print("=" * 60)
    print(f"Experiment ID: {EXPERIMENT_ID}")
    print(f"Mode: {mode}")
    
    # Auto-generate model path if not provided
    if model_path is None:
        model_path = f"models/xgboost_binary_classifier_{EXPERIMENT_ID}_{mode}.joblib"
    
    print(f"Using trained model: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ Trained model not found: {model_path}")
        print("Please train the predictor first using:")
        print(f"  python xgb_experiment.py train-{mode}")
        return None
    
    # Use default unseen test datasets if none provided
    if test_dataset_ids is None:
        test_dataset_ids = UNSEEN_TEST_DATASETS
    
    print(f"Testing on {len(test_dataset_ids)} unseen datasets: {test_dataset_ids}")
    print("These datasets were NOT used in knowledge base building to avoid data leakage")
    
    try:
        # Load the trained model
        print(f"\nLoading trained predictor...")
        model_data = joblib.load(model_path)
        
        # Print model info
        training_info = model_data.get('training_info', {})
        print(f"📊 Model trained on {training_info.get('n_training_samples', 'unknown')} samples")
        print(f"📊 Model R² score: {training_info.get('r2', 'unknown')}")
        
        print(f"\nStarting zero-shot evaluation...")
        print("-" * 60)
        
        results = []
        
        for dataset_id in test_dataset_ids:
            print(f"\n🔍 Testing on dataset {dataset_id}")
            
            try:
                # Fetch dataset
                data, target_name, dataset_name = fetch_open_ml_data(dataset_id)
                X, y = prepare_data(data, target_name)
                print(f"Dataset: {dataset_name}")
                print(f"Shape: {X.shape}")
                
                # Calculate meta-features
                X_df = convert_to_dataframe(X)
                meta_features = calculate_dataset_meta_parameters(X_df, y)
                
                # Prepare features for prediction
                feature_names = model_data['dataset_features']
                feature_vector = []
                
                for feature_name in feature_names:
                    if feature_name in meta_features:
                        feature_vector.append(float(meta_features[feature_name]))
                    else:
                        feature_vector.append(0.0)
                
                # Normalize features
                feature_vector = np.array(feature_vector, dtype=np.float64).reshape(1, -1)
                norm_params = model_data['normalization_params']
                
                for i, feature_name in enumerate(feature_names):
                    if feature_name in norm_params:
                        norm_info = norm_params[feature_name]
                        min_val = norm_info['min']
                        range_val = norm_info['range']
                        
                        if range_val > 0:
                            feature_vector[0, i] = (feature_vector[0, i] - min_val) / range_val
                        else:
                            feature_vector[0, i] = 0.0
                
                # Make prediction
                prediction = model_data['model'].predict(feature_vector)[0]
                
                # Convert to hyperparameters
                target_params = model_data['target_params']
                predicted_params = {}
                
                for i, param_name in enumerate(target_params):
                    clean_param_name = param_name.replace('params_', '')
                    
                    if i < len(prediction):
                        value = prediction[i]
                        
                        # Convert to appropriate type
                        if clean_param_name in ['n_estimators', 'max_depth']:
                            value = max(1, int(round(value)))
                        elif clean_param_name in ['learning_rate', 'subsample', 'colsample_bytree']:
                            value = max(0.01, min(1.0, float(value)))
                        
                        predicted_params[clean_param_name] = value
                
                print("Predicted hyperparameters:")
                for param, value in predicted_params.items():
                    print(f"  {param}: {value}")
                
                # Test the predicted hyperparameters
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                model = XGBClassifier(**predicted_params, random_state=42)
                model.fit(X_train, y_train)
                
                # Calculate AUC
                if hasattr(model, "predict_proba"):
                    y_proba = model.predict_proba(X_test)
                    if y_proba.shape[1] == 2:
                        auc_score = roc_auc_score(y_test, y_proba[:, 1])
                    else:
                        auc_score = roc_auc_score(y_test, y_proba, multi_class='ovr')
                else:
                    y_pred = model.predict(X_test)
                    auc_score = accuracy_score(y_test, y_pred)
                    print("Warning: Using accuracy instead of AUC")
                
                print(f"✅ Test AUC: {auc_score:.4f}")
                
                results.append({
                    'dataset_id': dataset_id,
                    'dataset_name': dataset_name,
                    'auc_score': auc_score,
                    'predicted_params': predicted_params,
                    'shape': X.shape
                })
                
            except Exception as e:
                print(f"❌ Error processing dataset {dataset_id}: {str(e)}")
                results.append({
                    'dataset_id': dataset_id,
                    'dataset_name': f"Dataset_{dataset_id}",
                    'auc_score': 0.0,
                    'error': str(e)
                })
        
        # Print summary
        print(f"\n{'='*60}")
        print("ZERO-SHOT PREDICTOR EVALUATION SUMMARY")
        print(f"{'='*60}")
        
        successful_tests = [r for r in results if 'error' not in r]
        failed_tests = [r for r in results if 'error' in r]
        
        if successful_tests:
            auc_scores = [r['auc_score'] for r in successful_tests]
            avg_auc = np.mean(auc_scores)
            std_auc = np.std(auc_scores)
            
            print(f"📊 Successful tests: {len(successful_tests)}/{len(test_dataset_ids)}")
            print(f"📊 Average AUC: {avg_auc:.4f} ± {std_auc:.4f}")
            print(f"📊 Best AUC: {max(auc_scores):.4f}")
            print(f"📊 Worst AUC: {min(auc_scores):.4f}")
            
            print(f"\nDetailed Results:")
            for result in successful_tests:
                print(f"  {result['dataset_id']:<6} {result['dataset_name']:<30} AUC: {result['auc_score']:.4f}")
        
        if failed_tests:
            print(f"\n❌ Failed tests: {len(failed_tests)}")
            for result in failed_tests:
                print(f"  {result['dataset_id']:<6} {result['dataset_name']:<30} ERROR: {result['error']}")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Error during predictor evaluation: {str(e)}")
        return None

def print_dataset_info():
    """Print information about the dataset collection."""
    print("\nDATASET COLLECTION INFORMATION")
    print("=" * 60)
    
    dataset_info = [
        (31, "credit-g", "1,000", "20", "2.333"),
        (38, "sick", "3,772", "29", "15.329"),
        (44, "spambase", "4,601", "57", "1.538"),
        (52, "trains", "10", "32", "1.000"),
        (151, "electricity", "45,312", "8", "1.355"),
        (179, "adult", "48,842", "14", "3.179"),
        (298, "coil2000", "9,822", "85", "15.761"),
        (846, "elevators", "16,599", "18", "2.236"),
        (1053, "jm1", "10,885", "21", "4.169"),
        (1112, "KDDCup09_churn", "50,000", "230", "12.617"),
        (1120, "MagicTelescope", "19,020", "10", "1.844"),
        (1128, "OVA_Breast", "1,545", "10,935", "3.491"),
        (1220, "Click_prediction_small", "39,948", "9", "4.938"),
        (40900, "Satellite", "5,100", "36", "67.000"),
        (45038, "road-safety", "111,762", "32", "1.000")
    ]
    
    print(f"{'ID':<6} {'Dataset Name':<25} {'n_samples':<10} {'n_features':<12} {'imbalance_ratio':<15}")
    print("-" * 75)
    
    for dataset_id, name, n_samples, n_features, imbalance in dataset_info:
        print(f"{dataset_id:<6} {name:<25} {n_samples:<10} {n_features:<12} {imbalance:<15}")
    
    print(f"\nTotal datasets: {len(dataset_info)}")
    print(f"Training datasets (test): {TEST_DATASET_COLLECTION}")
    print(f"Training datasets (full): {FULL_DATASET_COLLECTION}")
    print(f"Unseen test datasets: {UNSEEN_TEST_DATASETS}")

def main():
    """Main function for knowledge base building and predictor training."""
    print("XGBoost Knowledge Base Builder for Zero-Shot HPO")
    print("=" * 70)
    
    # Check command line arguments
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        mode = "info"
    
    if mode == "test":
        # Build knowledge base with test datasets (2 datasets, 10 iterations)
        print("Building TEST knowledge base...")
        kb_path, kb = build_knowledge_base(
            dataset_ids=TEST_DATASET_COLLECTION,
            n_iter=10,
            mode="test"
        )
        
    elif mode == "full":
        # Build knowledge base with all datasets (15 datasets, 20 iterations)
        print("Building FULL knowledge base...")
        kb_path, kb = build_knowledge_base(
            dataset_ids=FULL_DATASET_COLLECTION,
            n_iter=20,
            mode="full"
        )
        
    elif mode == "train-test":
        # Train predictor from test knowledge base
        print("Training zero-shot predictor from TEST knowledge base...")
        predictor = train_zero_shot_predictor(mode="test")
        
    elif mode == "train-full":
        # Train predictor from full knowledge base
        print("Training zero-shot predictor from FULL knowledge base...")
        predictor = train_zero_shot_predictor(mode="full")
        
    elif mode == "eval-test":
        # Evaluate trained predictor from test KB on unseen datasets
        print("Evaluating TEST predictor on unseen datasets...")
        results = test_zero_shot_predictor(mode="test")
        
    elif mode == "eval-full":
        # Evaluate trained predictor from full KB on unseen datasets
        print("Evaluating FULL predictor on unseen datasets...")
        results = test_zero_shot_predictor(mode="full")
        
    elif mode == "info":
        # Show dataset information
        print_dataset_info()
        print(f"\nExperiment ID: {EXPERIMENT_ID}")
        print("\nUsage:")
        print("  python xgb_experiment.py test         # Build test KB (2 datasets, 10 iter)")
        print("  python xgb_experiment.py full         # Build full KB (15 datasets, 20 iter)")
        print("  python xgb_experiment.py train-test   # Train predictor from test KB")
        print("  python xgb_experiment.py train-full   # Train predictor from full KB")
        print("  python xgb_experiment.py eval-test    # Evaluate test predictor on unseen data")
        print("  python xgb_experiment.py eval-full    # Evaluate full predictor on unseen data")
        print("  python xgb_experiment.py info         # Show dataset information")
        print("\nComplete Workflow:")
        print("  1. Build knowledge base: python xgb_experiment.py test")
        print("  2. Train predictor:      python xgb_experiment.py train-test")
        print("  3. Evaluate predictor:   python xgb_experiment.py eval-test")
        print("  4. Use in production:    from zerotune import ZeroTunePredictor")
        return
        
    else:
        print("Invalid mode. Use 'test', 'full', 'train-test', 'train-full', 'eval-test', 'eval-full', or 'info'")
        return
    
    if mode in ["test", "full"] and 'kb_path' in locals():
        if kb_path:
            print(f"\n🎉 Knowledge base building completed!")
            print(f"📁 Output: {kb_path}")
            print("\nNext step: Train a zero-shot predictor with:")
            print(f"  python xgb_experiment.py train-{mode}")
        else:
            print("\n❌ Knowledge base building failed.")

if __name__ == "__main__":
    main() 