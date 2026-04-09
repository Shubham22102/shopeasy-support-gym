#!/usr/bin/env python3
"""
Pre-submission validation script for ScalerX OpenEnv Hackathon.
Run this before submitting to ensure all graders work correctly.
"""

import sys
import math

def test_grader_format():
    """Test that all graders return correct format (score, info_dict)."""
    print("=" * 60)
    print("TEST 1: Grader Import & Format Validation")
    print("=" * 60)
    
    try:
        from grader.support_tasks import (
            SimpleRefundGrader, DeliveryTrackingGrader, KbPolicyQuestionGrader,
            CancellationRequestGrader, ExpiredReturnGrader, WrongItemSentGrader,
            DuplicateChargeGrader, PartialOrderGrader, DamagedItemGrader,
            AngryCustomerGrader, FraudRiskGrader, VipWarrantyClaimGrader
        )
    except ImportError as e:
        print(f"❌ FAILED: Cannot import graders - {e}")
        print("Make sure you're running from project root")
        sys.exit(1)
    
    graders = [
        SimpleRefundGrader, DeliveryTrackingGrader, KbPolicyQuestionGrader,
        CancellationRequestGrader, ExpiredReturnGrader, WrongItemSentGrader,
        DuplicateChargeGrader, PartialOrderGrader, DamagedItemGrader,
        AngryCustomerGrader, FraudRiskGrader, VipWarrantyClaimGrader
    ]
    
    passed_count = 0
    
    for GraderClass in graders:
        try:
            g = GraderClass()
            result = g.grade(world_state={
                "verified_facts": {"order_looked_up": True, "refund_processed": True},
                "resolution": "resolved",
                "step_count": 3,
                "max_steps": 10,
                "customer_mood": 0.5,
                "order": {"is_fraud_risk": False, "is_damaged": False, "status": "delivered"}
            })
            
            # Validate return type
            if not isinstance(result, tuple):
                print(f"❌ {GraderClass.__name__}: Returns {type(result)}, expected tuple")
                continue
                
            if len(result) != 2:
                print(f"❌ {GraderClass.__name__}: Returns {len(result)} items, expected 2 (score, info)")
                continue
            
            score, info = result
            
            # Validate score type and range
            if not isinstance(score, (int, float)):
                print(f"❌ {GraderClass.__name__}: Score is {type(score)}, expected float")
                continue
                
            if not (0 < score < 1):
                print(f"❌ {GraderClass.__name__}: Score {score} not in strict range (0, 1)")
                continue
            
            # Validate info dict
            if not isinstance(info, dict):
                print(f"❌ {GraderClass.__name__}: Info is {type(info)}, expected dict")
                continue
                
            required_keys = ["task_id", "score", "passed", "feedback", "score_range"]
            missing = [k for k in required_keys if k not in info]
            if missing:
                print(f"❌ {GraderClass.__name__}: Missing keys {missing} in info dict")
                continue
            
            # Validate score_range
            if not (isinstance(info["score_range"], (list, tuple)) and len(info["score_range"]) == 2):
                print(f"❌ {GraderClass.__name__}: score_range must be [min, max]")
                continue
                
            print(f"✅ {GraderClass.__name__}: score={score:.3f}, range={info['score_range']}")
            passed_count += 1
            
        except Exception as e:
            print(f"❌ {GraderClass.__name__}: Exception - {e}")
    
    print(f"\n📊 Result: {passed_count}/{len(graders)} graders passed")
    
    if passed_count < 3:
        print("❌ CRITICAL: Need at least 3 valid graders for submission!")
        return False
    elif passed_count < len(graders):
        print("⚠️  WARNING: Some graders failed, but minimum 3 passed")
        return True
    else:
        print("✅ All graders valid!")
        return True


def test_score_ranges():
    """Test that scores are strictly within (0, 1)."""
    print("\n" + "=" * 60)
    print("TEST 2: Score Range Validation (Strict 0 < score < 1)")
    print("=" * 60)
    
    from grader.support_tasks import FraudRiskGrader, SimpleRefundGrader
    
    # Test fraud risk (should be low but > 0)
    g = FraudRiskGrader()
    score, info = g.grade(world_state={
        "verified_facts": {"refund_processed": True},  # Wrong action
        "resolution": "unresolved",
        "step_count": 5,
        "max_steps": 10,
        "customer_mood": 0.0,
        "order": {"is_fraud_risk": True}
    })
    
    print(f"FraudRisk (wrong action): score={score}")
    assert 0 < score < 1, f"Score {score} out of range!"
    assert score < 0.1, "Fraud risk wrong action should score very low"
    
    # Test simple refund (should be high but < 1)
    g = SimpleRefundGrader()
    score, info = g.grade(world_state={
        "verified_facts": {"order_looked_up": True, "refund_processed": True},
        "resolution": "resolved",
        "step_count": 3,
        "max_steps": 10,
        "customer_mood": 0.5
    })
    
    print(f"SimpleRefund (correct): score={score}")
    assert 0 < score < 1, f"Score {score} out of range!"
    assert score > 0.8, "Correct simple refund should score high"
    
    print("✅ Score ranges valid!")


def test_inference_import():
    """Test that inference.py can be imported."""
    print("\n" + "=" * 60)
    print("TEST 3: Inference Script Validation")
    print("=" * 60)
    
    try:
        # Check file exists and has required functions
        import ast
        
        with open('inference.py', 'r') as f:
            content = f.read()
        
        # Parse to check for required functions
        tree = ast.parse(content)
        
        required_funcs = ['log_start', 'log_step', 'log_end', 'run_episode']
        found_funcs = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if node.name in required_funcs:
                    found_funcs.append(node.name)
        
        missing = set(required_funcs) - set(found_funcs)
        if missing:
            print(f"❌ Missing functions in inference.py: {missing}")
            return False
        
        # Check format strings
        if '[START] task=' not in content:
            print("⚠️  Warning: log_start format might be incorrect")
        
        if '[STEP] step=' not in content:
            print("⚠️  Warning: log_step format might be incorrect")
            
        if '[END] success=' not in content:
            print("⚠️  Warning: log_end format might be incorrect")
        
        print("✅ inference.py structure valid!")
        print(f"   Found functions: {found_funcs}")
        return True
        
    except Exception as e:
        print(f"❌ Error checking inference.py: {e}")
        return False


def main():
    print("🚀 ScalerX OpenEnv Pre-Submission Validator")
    print("=" * 60)
    
    results = []
    
    results.append(test_grader_format())
    results.append(test_score_ranges())
    results.append(test_inference_import())
    
    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)
    
    if all(results):
        print("✅ ALL TESTS PASSED - Ready for submission!")
        print("\nNext steps:")
        print("1. Build Docker image: docker build -t shopeasy-support .")
        print("2. Test locally: docker run -p 8000:8000 shopeasy-support")
        print("3. Push to Hugging Face Spaces")
        return 0
    else:
        print("❌ SOME TESTS FAILED - Fix before submitting!")
        return 1


if __name__ == "__main__":
    sys.exit(main())