#!/usr/bin/env python3
"""
Test script for verifying improved PromptBuilder V2.
Checks if ZeroTune Prior, Conditional Stats, and Correlations are correctly formatted.
"""

import sys
import os
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from zerotune.core.llm_surrogate.prompt_builder import PromptBuilder

def test_prompt_generation():
    print("Initializing PromptBuilder...")
    
    kb_path = "knowledge_base/kb_decision_tree_dt_kb_v1_test.json"
    if not os.path.exists(kb_path):
        print(f"Error: KB not found at {kb_path}")
        return

    builder = PromptBuilder(
        model_type="decision_tree",
        kb_path=kb_path,
        n_similar_datasets=1
    )
    
    # Mock data
    meta_features = {
        "n_samples": 1500,  # Medium dataset (matches KB data)
        "n_features": 20,
        "imbalance_ratio": 2.5,
        "n_highly_target_corr": 5,
        "avg_target_corr": 0.15
    }
    
    search_space = {
        "max_depth": (1, 20, "int"),
        "min_samples_leaf": (1, 20, "int"),
        "criterion": ["gini", "entropy"]
    }
    
    zt_prior = {
        "max_depth": 8,
        "min_samples_leaf": 5,
        "criterion": "gini"
    }
    
    print("\nGenerating Initial Prompt...")
    prompt = builder.build_initial_prompt(
        meta_features=meta_features,
        search_space=search_space,
        zt_prior=zt_prior
    )
    
    print("\n--- PROMPT PREVIEW ---")
    print(prompt)
    print("----------------------")
    
    # Verifications
    print("\nVerifying components:")
    
    # 1. ZeroTune Prior
    if "ZeroTune Zero-Shot Prediction" in prompt and "max_depth=8" in prompt:
        print("✅ ZeroTune Prior included")
    else:
        print("❌ ZeroTune Prior missing")
        
    # 2. Conditional Stats
    if "Historical Patterns for MEDIUM Datasets" in prompt:
        print("✅ Conditional Stats included (MEDIUM dataset detected)")
    else:
        print("❌ Conditional Stats missing")
        
    # 3. Correlations
    if "Relevant Parameter Correlations" in prompt:
        print("✅ Correlations included")
    else:
        print("⚠️ Correlations missing (might be due to insufficient data in test KB)")

if __name__ == "__main__":
    test_prompt_generation()
