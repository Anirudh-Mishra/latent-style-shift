#!/usr/bin/env python3
"""
Test script to verify U-ViT implementation is ready for training.
This script checks:
1. Model instantiation
2. Forward pass
3. Attention control integration
4. Source conditioning
5. Training compatibility
"""

import torch
import sys
from pathlib import Path

def test_uvit_backbone():
    """Test basic U-ViT backbone instantiation and forward pass."""
    print("\n=== Testing U-ViT Backbone ===")
    from uvit_backbone import UViTBackbone, UVIT_CONFIGS
    
    # Test all presets
    for preset in ["small", "mid", "large"]:
        print(f"\nTesting preset: {preset}")
        model = UViTBackbone.from_preset(
            preset,
            img_size=64,
            patch_size=2,
            in_chans=4,
            context_dim=768,
            source_conditioned=False,
        )
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {num_params / 1e6:.2f}M")
        
        # Test forward pass
        batch_size = 2
        x = torch.randn(batch_size, 4, 64, 64)
        timesteps = torch.randint(0, 1000, (batch_size,))
        context = torch.randn(batch_size, 77, 768)
        
        with torch.no_grad():
            output = model(x, timesteps, context)
        
        assert output.shape == x.shape, f"Output shape mismatch: {output.shape} vs {x.shape}"
        print(f"  ✓ Forward pass successful: {output.shape}")
    
    print("\n✓ U-ViT Backbone tests passed!")
    return True


def test_source_conditioning():
    """Test source-conditioned U-ViT."""
    print("\n=== Testing Source-Conditioned U-ViT ===")
    from uvit_backbone import UViTBackbone
    
    model = UViTBackbone.from_preset(
        "mid",
        img_size=64,
        patch_size=2,
        in_chans=4,
        context_dim=768,
        source_conditioned=True,
    )
    
    batch_size = 2
    x = torch.randn(batch_size, 4, 64, 64)
    source = torch.randn(batch_size, 4, 64, 64)
    timesteps = torch.randint(0, 1000, (batch_size,))
    context = torch.randn(batch_size, 77, 768)
    
    with torch.no_grad():
        output = model(x, timesteps, context, source_latent=source)
    
    assert output.shape == x.shape, f"Output shape mismatch: {output.shape} vs {x.shape}"
    print(f"  ✓ Source-conditioned forward pass successful: {output.shape}")
    
    # Test without source (should use zeros)
    with torch.no_grad():
        output2 = model(x, timesteps, context, source_latent=None)
    
    assert output2.shape == x.shape
    print(f"  ✓ Forward pass without source successful (uses zeros)")
    
    print("\n✓ Source conditioning tests passed!")
    return True


def test_uvit_adapter():
    """Test U-ViT adapter wrapper."""
    print("\n=== Testing U-ViT Adapter ===")
    from uvit_adapter import create_uvit_adapter, UViTAdapter
    
    adapter = create_uvit_adapter(preset="mid", source_conditioned=False)
    
    batch_size = 2
    sample = torch.randn(batch_size, 4, 64, 64)
    timestep = torch.tensor([500, 500])
    encoder_hidden_states = torch.randn(batch_size, 77, 768)
    
    with torch.no_grad():
        output = adapter(sample, timestep, encoder_hidden_states)
    
    assert hasattr(output, 'sample'), "Output should have 'sample' attribute"
    assert output.sample.shape == sample.shape
    print(f"  ✓ Adapter forward pass successful: {output.sample.shape}")
    
    # Test with source conditioning
    adapter_src = create_uvit_adapter(preset="mid", source_conditioned=True)
    source_latent = torch.randn(batch_size, 4, 64, 64)
    
    with torch.no_grad():
        output_src = adapter_src(sample, timestep, encoder_hidden_states, source_latent=source_latent)
    
    assert output_src.sample.shape == sample.shape
    print(f"  ✓ Source-conditioned adapter forward pass successful")
    
    print("\n✓ U-ViT Adapter tests passed!")
    return True


def test_attention_control():
    """Test attention control integration."""
    print("\n=== Testing Attention Control ===")
    from uvit_adapter import create_uvit_adapter, register_attention_control, unregister_attention_control, has_attention_processors
    
    adapter = create_uvit_adapter(preset="small", source_conditioned=False)
    
    # Create a simple controller
    class DummyController:
        def __init__(self):
            self.num_att_layers = 0
            self.calls = []
        
        def __call__(self, attn, is_cross, place_in_unet):
            self.calls.append((is_cross, place_in_unet))
            return attn
        
        def self_attn_forward(self, q, k, v, num_heads):
            return q, k, v
    
    controller = DummyController()
    
    # Register controller
    register_attention_control(adapter, controller)
    assert has_attention_processors(adapter), "Processors should be registered"
    print(f"  ✓ Registered {controller.num_att_layers} attention layers")
    
    # Test forward pass with controller
    batch_size = 1
    sample = torch.randn(batch_size, 4, 64, 64)
    timestep = torch.tensor([500])
    encoder_hidden_states = torch.randn(batch_size, 77, 768)
    
    with torch.no_grad():
        output = adapter(sample, timestep, encoder_hidden_states)
    
    assert len(controller.calls) > 0, "Controller should have been called"
    print(f"  ✓ Controller called {len(controller.calls)} times")
    
    # Unregister
    unregister_attention_control(adapter)
    assert not has_attention_processors(adapter), "Processors should be unregistered"
    print(f"  ✓ Processors unregistered successfully")
    
    print("\n✓ Attention control tests passed!")
    return True


def test_training_compatibility():
    """Test compatibility with training setup."""
    print("\n=== Testing Training Compatibility ===")
    from uvit_backbone import UViTBackbone
    from uvit_adapter import UViTAdapter
    
    # Create model
    backbone = UViTBackbone.from_preset(
        "small",
        img_size=64,
        patch_size=2,
        in_chans=4,
        context_dim=768,
        source_conditioned=True,
    )
    adapter = UViTAdapter(backbone)
    
    # Test gradient flow
    batch_size = 2
    sample = torch.randn(batch_size, 4, 64, 64, requires_grad=True)
    timestep = torch.tensor([500, 500])
    encoder_hidden_states = torch.randn(batch_size, 77, 768)
    source_latent = torch.randn(batch_size, 4, 64, 64)
    
    output = adapter(sample, timestep, encoder_hidden_states, source_latent=source_latent)
    loss = output.sample.mean()
    loss.backward()
    
    assert sample.grad is not None, "Gradients should flow to input"
    print(f"  ✓ Gradient flow verified")
    
    # Test optimizer step
    optimizer = torch.optim.AdamW(backbone.parameters(), lr=1e-4)
    optimizer.zero_grad()
    
    output = adapter(sample, timestep, encoder_hidden_states, source_latent=source_latent)
    loss = output.sample.mean()
    loss.backward()
    optimizer.step()
    
    print(f"  ✓ Optimizer step successful")
    
    # Test state dict save/load
    state_dict = backbone.state_dict()
    backbone2 = UViTBackbone.from_preset(
        "small",
        img_size=64,
        patch_size=2,
        in_chans=4,
        context_dim=768,
        source_conditioned=True,
    )
    backbone2.load_state_dict(state_dict)
    print(f"  ✓ State dict save/load successful")
    
    print("\n✓ Training compatibility tests passed!")
    return True


def test_dtype_handling():
    """Test mixed precision and dtype handling."""
    print("\n=== Testing Dtype Handling ===")
    from uvit_adapter import create_uvit_adapter
    
    # Test float32
    adapter_fp32 = create_uvit_adapter(preset="small")
    adapter_fp32 = adapter_fp32.to(dtype=torch.float32)
    
    sample = torch.randn(1, 4, 64, 64, dtype=torch.float32)
    timestep = torch.tensor([500])
    context = torch.randn(1, 77, 768, dtype=torch.float32)
    
    with torch.no_grad():
        output = adapter_fp32(sample, timestep, context)
    
    assert output.sample.dtype == torch.float32
    print(f"  ✓ Float32 forward pass successful")
    
    # Test float16 if CUDA available
    if torch.cuda.is_available():
        adapter_fp16 = create_uvit_adapter(preset="small")
        adapter_fp16 = adapter_fp16.to(dtype=torch.float16, device='cuda')
        
        sample = torch.randn(1, 4, 64, 64, dtype=torch.float16, device='cuda')
        timestep = torch.tensor([500], device='cuda')
        context = torch.randn(1, 77, 768, dtype=torch.float16, device='cuda')
        
        with torch.no_grad():
            output = adapter_fp16(sample, timestep, context)
        
        assert output.sample.dtype == torch.float16
        print(f"  ✓ Float16 forward pass successful (CUDA)")
    else:
        print(f"  ⊘ Skipping float16 test (CUDA not available)")
    
    print("\n✓ Dtype handling tests passed!")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("U-ViT Implementation Verification")
    print("=" * 60)
    
    tests = [
        ("U-ViT Backbone", test_uvit_backbone),
        ("Source Conditioning", test_source_conditioning),
        ("U-ViT Adapter", test_uvit_adapter),
        ("Attention Control", test_attention_control),
        ("Training Compatibility", test_training_compatibility),
        ("Dtype Handling", test_dtype_handling),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, success, None))
        except Exception as e:
            print(f"\n✗ {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False, str(e)))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    for name, success, error in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {name}")
        if error:
            print(f"       Error: {error}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The U-ViT implementation is ready for training.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
