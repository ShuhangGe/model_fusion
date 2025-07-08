#!/usr/bin/env python3
# coding=utf-8
# Copyright 2025 The ZhipuAI Inc. team and HuggingFace Inc. team. All rights reserved.

"""
Test runner for GLM4V Vision Encoder

Run this script from the project root to test the standalone GLM4V vision encoder.

Usage:
    python run_tests.py
"""

import sys
import os

def main():
    """Run the GLM4V vision encoder test suite."""
    print("GLM4V Vision Encoder - Test Runner")
    print("=" * 50)
    
    # Add current directory to Python path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    try:
        # Import and run the test suite
        from model.test_vision import run_all_tests
        
        print("Starting test suite...\n")
        success = run_all_tests()
        
        if success:
            print("\n🎉 All tests passed! GLM4V Vision Encoder is ready to use.")
            return 0
        else:
            print("\n❌ Some tests failed. Please check the output above for details.")
            return 1
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\nMake sure you're running this script from the project root directory.")
        print("The 'model' directory should be in the same directory as this script.")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 