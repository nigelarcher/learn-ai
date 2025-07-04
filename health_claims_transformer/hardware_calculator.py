#!/usr/bin/env python3

def calculate_memory_requirements(vocab_size, d_model, n_heads, n_layers, seq_length=512, batch_size=32, training=True):
    """Calculate exact memory requirements for transformer."""
    
    print(f"🧮 Memory Calculator")
    print(f"Config: vocab={vocab_size:,} | d_model={d_model} | heads={n_heads} | layers={n_layers}")
    print("=" * 70)
    
    # 1. Embedding layer memory
    embedding_params = vocab_size * d_model
    embedding_memory = embedding_params * 4  # 4 bytes per float32
    print(f"📚 Embeddings: {embedding_params:,} params = {embedding_memory / (1024**2):.1f}MB")
    
    # 2. Attention layers memory
    attention_params_per_layer = 4 * (d_model * d_model)  # W_q, W_k, W_v, W_o
    total_attention_params = attention_params_per_layer * n_layers
    attention_memory = total_attention_params * 4
    print(f"🎯 Attention: {total_attention_params:,} params = {attention_memory / (1024**2):.1f}MB")
    
    # 3. Feed-forward layers memory  
    d_ff = d_model * 4  # Standard FFN is 4x model dimension
    ffn_params_per_layer = (d_model * d_ff) + (d_ff * d_model)  # Two linear layers
    total_ffn_params = ffn_params_per_layer * n_layers
    ffn_memory = total_ffn_params * 4
    print(f"🔄 Feed-Forward: {total_ffn_params:,} params = {ffn_memory / (1024**2):.1f}MB")
    
    # 4. Layer normalization
    norm_params = d_model * 2 * n_layers * 2  # gamma, beta for each norm layer
    norm_memory = norm_params * 4
    print(f"📏 Layer Norm: {norm_params:,} params = {norm_memory / (1024**2):.1f}MB")
    
    # 5. Output projection
    output_params = d_model * vocab_size
    output_memory = output_params * 4
    print(f"📤 Output: {output_params:,} params = {output_memory / (1024**2):.1f}MB")
    
    # Total model weights
    total_params = embedding_params + total_attention_params + total_ffn_params + norm_params + output_params
    model_memory = total_params * 4
    print(f"\n🏗️  TOTAL MODEL: {total_params:,} params = {model_memory / (1024**2):.1f}MB")
    
    # 6. Activations during forward pass
    activation_memory_per_layer = batch_size * seq_length * d_model * 4
    total_activation_memory = activation_memory_per_layer * n_layers
    print(f"⚡ Activations: {total_activation_memory / (1024**2):.1f}MB")
    
    # 7. Attention scores memory (the O(n²) killer!)
    attention_scores_memory = batch_size * n_heads * seq_length * seq_length * 4 * n_layers
    print(f"🎭 Attention Scores: {attention_scores_memory / (1024**2):.1f}MB")
    
    # 8. Training overhead (gradients + optimizer states)
    if training:
        gradient_memory = model_memory  # Same size as model
        optimizer_memory = model_memory * 2  # Adam stores momentum + variance
        training_overhead = gradient_memory + optimizer_memory
        print(f"🎓 Training Overhead: {training_overhead / (1024**2):.1f}MB")
    else:
        training_overhead = 0
    
    # Total memory needed
    total_memory = model_memory + total_activation_memory + attention_scores_memory + training_overhead
    
    print(f"\n💾 TOTAL VRAM NEEDED: {total_memory / (1024**2):.1f}MB ({total_memory / (1024**3):.2f}GB)")
    
    return total_memory

def gpu_recommendations():
    """Show what GPU you need for different model sizes."""
    
    print("\n🎮 GPU Recommendations")
    print("=" * 70)
    
    configurations = [
        # name, vocab_size, d_model, n_heads, n_layers
        ("Tiny (Learning)", 1000, 128, 4, 2),
        ("Small (Demo)", 5000, 256, 8, 4), 
        ("Medium (Startup)", 15000, 512, 8, 6),
        ("Large (Production)", 25000, 768, 12, 8),
        ("Huge (Research)", 50000, 1024, 16, 12),
        ("Massive (GPT-style)", 100000, 2048, 32, 24)
    ]
    
    gpu_specs = {
        "RTX 3060": 12,
        "RTX 3070": 8, 
        "RTX 3080": 10,
        "RTX 3090": 24,
        "RTX 4070": 12,
        "RTX 4080": 16,
        "RTX 4090": 24,
        "V100": 32,
        "A100": 80,
        "H100": 80
    }
    
    print(f"{'Model Size':<20} {'VRAM Needed':<12} {'Recommended GPU':<15} {'Can Train?'}")
    print("-" * 70)
    
    for name, vocab, d_model, heads, layers in configurations:
        memory_needed = calculate_memory_requirements(vocab, d_model, heads, layers, 
                                                    seq_length=512, batch_size=8, training=True)
        memory_gb = memory_needed / (1024**3)
        
        # Find suitable GPU
        suitable_gpus = [(gpu, vram) for gpu, vram in gpu_specs.items() if vram >= memory_gb]
        if suitable_gpus:
            recommended = min(suitable_gpus, key=lambda x: x[1])
            can_train = "✅ Yes" if memory_gb < recommended[1] * 0.8 else "⚠️  Tight"
        else:
            recommended = ("None available", 0)
            can_train = "❌ No"
        
        print(f"{name:<20} {memory_gb:>8.1f}GB    {recommended[0]:<15} {can_train}")

def scaling_bottlenecks():
    """Show what happens when you increase each parameter."""
    
    print("\n\n📈 Scaling Bottlenecks: What Eats Your VRAM")
    print("=" * 70)
    
    base_config = (10000, 512, 8, 6)  # vocab, d_model, heads, layers
    
    print("🔍 Impact of doubling each parameter:")
    print(f"{'Parameter':<15} {'Base Memory':<12} {'2x Memory':<12} {'Increase'}")
    print("-" * 55)
    
    # Base memory
    base_memory = calculate_memory_requirements(*base_config, seq_length=512, batch_size=8, training=False)
    base_gb = base_memory / (1024**3)
    
    # Test doubling each parameter
    tests = [
        ("vocab_size", (20000, 512, 8, 6)),
        ("d_model", (10000, 1024, 8, 6)), 
        ("n_heads", (10000, 512, 16, 6)),
        ("n_layers", (10000, 512, 8, 12))
    ]
    
    for param_name, config in tests:
        new_memory = calculate_memory_requirements(*config, seq_length=512, batch_size=8, training=False)
        new_gb = new_memory / (1024**3)
        increase = new_gb / base_gb
        
        print(f"{param_name:<15} {base_gb:>8.1f}GB    {new_gb:>8.1f}GB    {increase:>5.1f}x")
    
    print("\n💡 Key Insights:")
    print("   • d_model has QUADRATIC impact (most expensive!)")
    print("   • vocab_size has LINEAR impact") 
    print("   • n_heads has LINEAR impact")
    print("   • n_layers has LINEAR impact")
    print("   • Sequence length has QUADRATIC impact on attention!")

def practical_advice():
    """Practical advice for hardware selection."""
    
    print("\n\n🎯 Practical Hardware Advice")
    print("=" * 70)
    
    use_cases = {
        "Learning/Experimentation": {
            "config": "1K vocab, 128 d_model, 4 heads, 2 layers",
            "vram_needed": "1-2GB",
            "gpu_options": "RTX 3060, RTX 4060, GTX 1080",
            "cost": "$200-400"
        },
        "Startup MVP": {
            "config": "5K vocab, 256 d_model, 8 heads, 4 layers", 
            "vram_needed": "4-6GB",
            "gpu_options": "RTX 3070, RTX 4070",
            "cost": "$400-600"
        },
        "Production (nib scale)": {
            "config": "15K vocab, 512 d_model, 8 heads, 6 layers",
            "vram_needed": "12-16GB", 
            "gpu_options": "RTX 3090, RTX 4080, V100",
            "cost": "$800-1500"
        },
        "Research/Large Scale": {
            "config": "50K vocab, 1024 d_model, 16 heads, 12 layers",
            "vram_needed": "32-80GB",
            "gpu_options": "A100, H100",
            "cost": "$5000-15000"
        }
    }
    
    for use_case, specs in use_cases.items():
        print(f"\n🏢 {use_case}")
        print(f"   Config: {specs['config']}")
        print(f"   VRAM needed: {specs['vram_needed']}")
        print(f"   GPU options: {specs['gpu_options']}")
        print(f"   Cost range: {specs['cost']}")
    
    print(f"\n💰 Cost vs Performance Trade-offs:")
    print(f"   • 2GB VRAM: Can run tiny models for learning")
    print(f"   • 8GB VRAM: Good for prototyping and small production")  
    print(f"   • 16GB VRAM: Production ready for most use cases")
    print(f"   • 24GB VRAM: High-end production, complex models")
    print(f"   • 80GB VRAM: Research, massive models, large batches")

if __name__ == "__main__":
    # Demo calculations
    print("🚀 Transformer Hardware Requirements Calculator")
    print("=" * 70)
    
    # Calculate our demo model
    calculate_memory_requirements(1000, 128, 4, 2, seq_length=256, batch_size=8, training=False)
    print("\n" + "="*70)
    
    # Calculate production model  
    calculate_memory_requirements(15000, 512, 8, 6, seq_length=512, batch_size=32, training=True)
    
    gpu_recommendations()
    scaling_bottlenecks() 
    practical_advice()