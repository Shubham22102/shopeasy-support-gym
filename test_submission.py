#!/usr/bin/env python3
"""
Pre-submission validation script for ScalerX OpenEnv Hackathon.
"""

import sys

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
        return False
    
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
            
            if not isinstance(result, tuple) or len(result) != 2:
                print(f"❌ {GraderClass.__name__}: Invalid return format")
                continue
            
            score, info = result
            
            if not isinstance(score, (int, float)) or not (0 < score < 1):
                print(f"❌ {GraderClass.__name__}: Score {score} not in (0, 1)")
                continue
            
            if not isinstance(info, dict) or "task_id" not in info:
                print(f"❌ {GraderClass.__name__}: Invalid info dict")
                continue
            
            print(f"✅ {GraderClass.__name__}: score={score:.3f}")
            passed_count += 1
            
        except Exception as e:
            print(f"❌ {GraderClass.__name__}: {e}")
    
    print(f"\n📊 {passed_count}/{len(graders)} graders passed")
    return passed_count >= 3  # Return True if minimum 3 passed


def test_score_ranges():
    """Test edge cases for score ranges."""
    print("\n" + "=" * 60)
    print("TEST 2: Score Range Edge Cases")
    print("=" * 60)
    
    from grader.support_tasks import FraudRiskGrader, SimpleRefundGrader
    
    # Test worst case (should be > 0)
    g = FraudRiskGrader()
    score, _ = g.grade(world_state={
        "verified_facts": {"refund_processed": True},
        "resolution": "unresolved",
        "step_count": 5,
        "max_steps": 10,
        "customer_mood": 0.0,
        "order": {"is_fraud_risk": True}
    })
    
    print(f"FraudRisk (penalized): score={score}")
    assert 0 < score < 1, f"Score {score} out of range!"
    
    # Test best case (should be < 1)
    g = SimpleRefundGrader()
    score, _ = g.grade(world_state={
        "verified_facts": {"order_looked_up": True, "refund_processed": True},
        "resolution": "resolved",
        "step_count": 2,
        "max_steps": 10,
        "customer_mood": 0.5
    })
    
    print(f"SimpleRefund (optimal): score={score}")
    assert 0 < score < 1, f"Score {score} out of range!"
    
    print("✅ Score ranges valid!")
    return True  # FIXED: Added return True


def test_inference_structure():
    """Test inference.py structure."""
    print("\n" + "=" * 60)
    print("TEST 3: Inference Script Validation")
    print("=" * 60)
    
    try:
        import ast
        with open('inference.py', 'r') as f:
            tree = ast.parse(f.read())
        
        funcs = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        required = ['log_start', 'log_step', 'log_end', 'run_episode']
        
        for func in required:
            if func not in funcs:
                print(f"❌ Missing function: {func}")
                return False
        
        print(f"✅ Found all required functions: {required}")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    print("🚀 ScalerX OpenEnv Pre-Submission Validator\n")
    
    # Run all tests
    test1 = test_grader_format()
    test2 = test_score_ranges()  # Now properly returns True/False
    test3 = test_inference_structure()
    
    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)
    
    if test1 and test2 and test3:
        print("✅ ALL TESTS PASSED - Ready for submission!")
        print("\nNext steps:")
        print("1. docker build -t shopeasy-support .")
        print("2. Test: docker run -p 8000:8000 shopeasy-support")
        print("3. Push to Hugging Face Spaces")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())