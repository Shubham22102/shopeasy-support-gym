#!/usr/bin/env python3
"""Pre-submission validation script"""

import os
import sys

def check_files():
    required = ['inference.py', 'openenv.yaml', 'Dockerfile', 'README.md']
    for f in required:
        if not os.path.exists(f):
            print(f"❌ MISSING: {f}")
            return False
    print("✅ All required files present")
    return True

def check_inference_format():
    """Simulate running inference and check output format"""
    import subprocess
    import re
    
    # Set dummy env vars
    env = os.environ.copy()
    env['HF_TOKEN'] = 'dummy'
    env['OPENAI_API_KEY'] = 'dummy'
    
    try:
        result = subprocess.run(
            ['python', 'inference.py'],
            capture_output=True,
            text=True,
            timeout=30,
            env=env
        )
        
        lines = result.stdout.strip().split('\n')
        
        # Check START line
        if not re.match(r'\[START\]task=\w+env=\w+model=\w+', lines[0]):
            print("❌ START line format incorrect")
            return False
            
        # Check END line exists
        if not any('[END]' in line for line in lines):
            print("❌ Missing [END] line")
            return False
            
        print("✅ Output format valid")
        return True
        
    except Exception as e:
        print(f"❌ Inference execution failed: {e}")
        return False

def check_env_interface():
    """Check that environment implements required methods"""
    from server.Customer_Support_Gym_2_environment import SupportEnvironment
    from models import SupportAction, SupportObservation

    env = SupportEnvironment()

    # Check methods exist
    assert hasattr(env, 'reset'), "Missing reset()"
    assert hasattr(env, 'step'), "Missing step()"
    assert hasattr(env, 'state'), "Missing state property"

    # Pydantic v2 stores field metadata in model_fields, not as class attrs
    action_fields = SupportAction.model_fields
    obs_fields    = SupportObservation.model_fields

    assert 'action_type' in action_fields, "SupportAction missing field: action_type"
    assert 'done'        in obs_fields,    "SupportObservation missing field: done"

    # Live smoke-test: actually reset the environment
    obs = env.reset(task_id="simple_refund")
    assert obs is not None,          "reset() returned None"
    assert obs.done is False,        "New episode should not be done"
    assert obs.task_id != "",        "task_id must be set after reset"
    assert obs.ticket_id != "",      "ticket_id must be set after reset"
    assert obs.customer_message != "", "Opening message must not be empty"

    print("✅ Environment interface valid")
    print(f"   task_id={obs.task_id!r}  ticket_id={obs.ticket_id!r}")
    print(f"   opening: {obs.customer_message[:60]!r}...")
    return True

if __name__ == "__main__":
    checks = [
        check_files(),
        check_env_interface(),
        # check_inference_format(),  # Uncomment after setting up test env
    ]
    
    if all(checks):
        print("\n🎉 All checks passed! Ready for submission.")
    else:
        print("\n⚠️ Fix issues before submitting.")
        sys.exit(1)