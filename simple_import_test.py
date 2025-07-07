#!/usr/bin/env python3

"""
Simple import test to debug the Qwen import issues step by step.
"""

import sys
import os

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

print("=" * 50)
print("Simple Import Test")
print("=" * 50)
print(f"Working directory: {current_dir}")
print(f"Python path includes: {current_dir}")

# Test 1: Check if transformers directory exists
transformers_path = os.path.join(current_dir, "transformers")
print(f"✓ Transformers directory exists: {os.path.exists(transformers_path)}")

# Test 2: Try to import transformers
try:
    import transformers
    print(f"✓ Transformers import successful")
    print(f"  Transformers path: {transformers.__file__}")
except Exception as e:
    print(f"✗ Transformers import failed: {e}")

# Test 3: Try to import specific transformers modules
modules_to_test = [
    "transformers.utils",
    "transformers.activations", 
    "transformers.modeling_utils",
    "transformers.configuration_utils"
]

for module in modules_to_test:
    try:
        __import__(module)
        print(f"✓ {module} import successful")
    except Exception as e:
        print(f"✗ {module} import failed: {e}")

# Test 4: Try to import qwen configuration only
try:
    from qwen.qwen3.configuration_qwen3 import Qwen3Config
    print("✓ Qwen3Config import successful")
except Exception as e:
    print(f"✗ Qwen3Config import failed: {e}")

# Test 5: Try to import qwen modeling
try:
    from qwen.qwen3.modeling_qwen3 import Qwen3ForCausalLM
    print("✓ Qwen3ForCausalLM import successful")
except Exception as e:
    print(f"✗ Qwen3ForCausalLM import failed: {e}")

print("=" * 50)
print("Import test complete") 