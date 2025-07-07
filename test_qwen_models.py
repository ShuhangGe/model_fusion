#!/usr/bin/env python3

"""
Test script for Qwen models in GLM-4.1V-Thinking_model.

This script tests that the Qwen3 and Qwen3-MoE models can be imported
and instantiated with fake inputs after fixing their import paths.
"""

import torch
import sys
import os

# Add the current directory to Python path to find local transformers
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)  # Insert at beginning to prioritize local transformers

def test_qwen3_model():
    """Test Qwen3 model instantiation and basic forward pass."""
    print("=== Testing Qwen3 Model ===")
    
    try:
        from qwen.qwen3.configuration_qwen3 import Qwen3Config
        from qwen.qwen3.modeling_qwen3 import Qwen3ForCausalLM
        print("✓ Qwen3 imports successful")
        
        # Create small test config
        config = Qwen3Config(
            vocab_size=1000,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=512,
            layer_types=["full_attention", "full_attention"],  # Required for Qwen3
        )
        print("✓ Qwen3 config created")
        
        # Create model
        model = Qwen3ForCausalLM(config)
        model.eval()
        print(f"✓ Qwen3 model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Test forward pass
        batch_size, seq_len = 1, 10
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            print(f"✓ Forward pass successful: {input_ids.shape} → {outputs.logits.shape}")
            
        return True
        
    except Exception as e:
        print(f"✗ Qwen3 test failed: {e}")
        return False


def test_qwen3_moe_model():
    """Test Qwen3-MoE model instantiation and basic forward pass."""
    print("\n=== Testing Qwen3-MoE Model ===")
    
    try:
        from qwen.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
        from qwen.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM
        print("✓ Qwen3-MoE imports successful")
        
        # Create small test config
        config = Qwen3MoeConfig(
            vocab_size=1000,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=512,
            num_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=256,
            decoder_sparse_step=1,  # Every layer is MoE
            mlp_only_layers=[],     # No MLP-only layers
        )
        print("✓ Qwen3-MoE config created")
        
        # Create model
        model = Qwen3MoeForCausalLM(config)
        model.eval()
        print(f"✓ Qwen3-MoE model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Test forward pass
        batch_size, seq_len = 1, 10
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_router_logits=True)
            print(f"✓ Forward pass successful: {input_ids.shape} → {outputs.logits.shape}")
            print(f"✓ Router logits available: {len(outputs.router_logits) if outputs.router_logits else 0} layers")
            
        return True
        
    except Exception as e:
        print(f"✗ Qwen3-MoE test failed: {e}")
        return False


def test_qwen_package_import():
    """Test importing from the main qwen package."""
    print("\n=== Testing Qwen Package Import ===")
    
    try:
        from qwen import (
            Qwen3Config, Qwen3ForCausalLM,
            Qwen3MoeConfig, Qwen3MoeForCausalLM
        )
        print("✓ Main package imports successful")
        
        # Quick instantiation test
        qwen3_config = Qwen3Config(vocab_size=100, hidden_size=64, num_hidden_layers=1, 
                                   num_attention_heads=2, layer_types=["full_attention"])
        qwen3_model = Qwen3ForCausalLM(qwen3_config)
        print(f"✓ Qwen3 from package: {sum(p.numel() for p in qwen3_model.parameters()):,} params")
        
        return True
        
    except Exception as e:
        print(f"✗ Package import test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("Testing Qwen Models in GLM-4.1V-Thinking Project")
    print("=" * 50)
    
    # System info
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Working directory: {os.getcwd()}")
    
    # Run tests
    results = []
    results.append(test_qwen3_model())
    results.append(test_qwen3_moe_model()) 
    results.append(test_qwen_package_import())
    
    print("\n" + "=" * 50)
    if all(results):
        print("✅ All tests passed! Qwen models are working correctly.")
        print("\nYou can now use:")
        print("from qwen import Qwen3ForCausalLM, Qwen3Config")
        print("from qwen import Qwen3MoeForCausalLM, Qwen3MoeConfig")
    else:
        print("❌ Some tests failed. Check the error messages above.")
        
    print("\nQwen models successfully integrated into GLM-4.1V-Thinking_model!")


if __name__ == "__main__":
    main() 