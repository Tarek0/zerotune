#!/usr/bin/env python3
"""
Test script for verifying improved PromptBuilder V2.
Checks if ZeroTune Prior, Conditional Stats, Correlations, and Adaptive Strategy are correctly formatted.
"""

import sys
import os
from pathlib import Path
from dataclasses import dataclass

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from zerotune.core.llm_surrogate.prompt_builder import PromptBuilder


@dataclass
class MockTrial:
    """Mock Optuna trial for testing sequential prompts."""
    number: int
    value: float
    params: dict


def test_initial_prompt():
    """Test initial prompt generation."""
    print("=" * 60)
    print("TEST 1: Initial Prompt")
    print("=" * 60)
    
    kb_path = "knowledge_base/kb_decision_tree_dt_kb_v1_test.json"
    if not os.path.exists(kb_path):
        print(f"Error: KB not found at {kb_path}")
        return False

    builder = PromptBuilder(
        model_type="decision_tree",
        kb_path=kb_path,
        n_similar_datasets=1
    )
    
    meta_features = {
        "n_samples": 1500,
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
    
    zt_prior = {"max_depth": 8, "min_samples_leaf": 5, "criterion": "gini"}
    
    prompt = builder.build_initial_prompt(
        meta_features=meta_features,
        search_space=search_space,
        zt_prior=zt_prior
    )
    
    print("\n--- INITIAL PROMPT (truncated) ---")
    print(prompt[:2000] + "\n...[truncated]...")
    print("-----------------------------------")
    
    # Verifications
    print("\nVerifying Initial Prompt components:")
    passed = True
    
    if "ZeroTune Zero-Shot Prediction" in prompt and "max_depth=8" in prompt:
        print("✅ ZeroTune Prior included")
    else:
        print("❌ ZeroTune Prior missing")
        passed = False
        
    if "Historical Patterns for MEDIUM Datasets" in prompt:
        print("✅ Conditional Stats included")
    else:
        print("❌ Conditional Stats missing")
        passed = False
    
    return passed


def test_sequential_prompt():
    """Test sequential prompt with adaptive strategy."""
    print("\n" + "=" * 60)
    print("TEST 2: Sequential Prompt with Adaptive Strategy")
    print("=" * 60)
    
    kb_path = "knowledge_base/kb_decision_tree_dt_kb_v1_test.json"
    builder = PromptBuilder(
        model_type="decision_tree",
        kb_path=kb_path,
        n_similar_datasets=1
    )
    
    meta_features = {
        "n_samples": 1500,
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
    
    zt_prior = {"max_depth": 8, "min_samples_leaf": 5, "criterion": "gini"}
    
    # Mock completed trials
    mock_trials = [
        MockTrial(number=0, value=0.72, params={"max_depth": 5, "min_samples_leaf": 10, "criterion": "gini"}),
        MockTrial(number=1, value=0.75, params={"max_depth": 8, "min_samples_leaf": 5, "criterion": "gini"}),
        MockTrial(number=2, value=0.73, params={"max_depth": 12, "min_samples_leaf": 3, "criterion": "entropy"}),
    ]
    
    # Test with HIGH confidence (similar to KB)
    prompt_high = builder.build_sequential_prompt(
        meta_features=meta_features,
        search_space=search_space,
        completed_trials=mock_trials,
        zt_prior=zt_prior,
        kb_confidence=0.90
    )
    
    # Test with LOW confidence (dissimilar to KB)
    prompt_low = builder.build_sequential_prompt(
        meta_features=meta_features,
        search_space=search_space,
        completed_trials=mock_trials,
        zt_prior=zt_prior,
        kb_confidence=0.45
    )
    
    print("\nVerifying Sequential Prompt components:")
    passed = True
    
    # Check HIGH confidence strategy
    if "HIGH CONFIDENCE" in prompt_high and "Prioritize **EXPLOIT**" in prompt_high:
        print("✅ HIGH confidence strategy recommendation works")
    else:
        print("❌ HIGH confidence strategy missing")
        passed = False
        
    # Check LOW confidence strategy
    if "LOW CONFIDENCE" in prompt_low and "Prioritize **EXPLORE**" in prompt_low:
        print("✅ LOW confidence strategy recommendation works")
    else:
        print("❌ LOW confidence strategy missing")
        passed = False
    
    # Show strategy section preview
    if "Recommended Strategy" in prompt_high:
        start = prompt_high.find("## Recommended Strategy")
        end = prompt_high.find("## Task", start)
        print("\n--- Strategy Section (HIGH confidence) ---")
        print(prompt_high[start:end])
    
    return passed


def main():
    print("\n🔬 LLM Surrogate V2 Prompt Verification\n")
    
    test1_passed = test_initial_prompt()
    test2_passed = test_sequential_prompt()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Initial Prompt: {'✅ PASS' if test1_passed else '❌ FAIL'}")
    print(f"Sequential Prompt (Adaptive Strategy): {'✅ PASS' if test2_passed else '❌ FAIL'}")
    
    if test1_passed and test2_passed:
        print("\n✅ All V2 improvements verified successfully!")
    else:
        print("\n❌ Some tests failed. Review output above.")


if __name__ == "__main__":
    main()
