#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from tokenizer import MedicalTokenizer

def demonstrate_vocab_size_impact():
    """Show how vocab_size affects medical term coverage."""
    
    print("🔤 Vocabulary Size Impact on Medical Coverage")
    print("=" * 60)
    
    # Medical claims with varying complexity
    test_claims = [
        # Basic claim
        "Diabetes patient prescribed Metformin 500mg",
        
        # Common claim  
        "Hypertension treated with Lisinopril 10mg daily",
        
        # Complex claim
        "Rheumatoid arthritis patient given Adalimumab 40mg biweekly",
        
        # Cutting-edge claim
        "Alzheimer's patient prescribed Leqembi 10mg monthly",
        
        # Very rare condition
        "Fabry disease treated with Agalsidase beta infusion"
    ]
    
    # Test different vocabulary sizes
    vocab_sizes = [500, 2000, 10000]
    
    for vocab_size in vocab_sizes:
        print(f"\n📚 Testing vocab_size = {vocab_size}")
        print("-" * 40)
        
        tokenizer = MedicalTokenizer(vocab_size=vocab_size)
        
        # Build vocabulary from basic medical examples
        basic_examples = [
            "Patient diabetes metformin insulin",
            "Hypertension blood pressure medication",
            "Office visit procedure code billing"
        ]
        tokenizer.build_vocabulary(basic_examples)
        
        # Test coverage
        total_tokens = 0
        unk_tokens = 0
        
        for claim in test_claims:
            tokens = tokenizer._tokenize_medical_terms(claim)
            encoded = tokenizer.encode(claim, max_length=20)
            
            # Count unknowns
            claim_total = len(tokens)
            claim_unks = sum(1 for t in tokens if t not in tokenizer.word_to_id)
            
            total_tokens += claim_total
            unk_tokens += claim_unks
            
            # Show first claim in detail
            if claim == test_claims[0]:
                print(f"   '{claim}'")
                print(f"   Tokens: {tokens}")
                print(f"   Unknown: {claim_unks}/{claim_total}")
        
        coverage = (total_tokens - unk_tokens) / total_tokens * 100
        print(f"   Overall coverage: {coverage:.1f}%")
        print(f"   Memory usage: ~{vocab_size * 4 / 1024:.1f}KB")

def demonstrate_d_model_impact():
    """Show how d_model affects representation richness."""
    
    print("\n\n🧠 d_model Impact on Word Understanding")
    print("=" * 60)
    
    # Simulate different d_model sizes
    d_model_sizes = [64, 256, 1024]
    
    medical_terms = [
        "metformin",
        "diabetes", 
        "hypertension",
        "500mg"
    ]
    
    for d_model in d_model_sizes:
        print(f"\n🎯 d_model = {d_model}")
        print("-" * 30)
        
        print(f"   Each word represented by {d_model} numbers")
        print(f"   Memory per word: {d_model * 4} bytes")
        
        # Simulate what the model could "understand"
        if d_model == 64:
            understanding = "Basic: diabetes drug, dosage"
        elif d_model == 256:
            understanding = "Detailed: diabetes drug + mechanism + dosing + side effects"
        else:  # 1024
            understanding = "Expert: diabetes drug + mechanism + dosing + side effects + interactions + contraindications + patient factors"
        
        print(f"   Understanding level: {understanding}")
        
        # Show relationship capacity
        relationship_capacity = d_model // 16  # Rough estimate
        print(f"   Can track ~{relationship_capacity} different relationships")
        
        # Memory calculation
        total_memory_mb = (d_model * 10000 * 4) / (1024 * 1024)  # 10K vocab
        print(f"   Total embedding memory: {total_memory_mb:.1f}MB")

def demonstrate_combined_effect():
    """Show how vocab_size and d_model work together."""
    
    print("\n\n⚖️  Combined Effect: The Sweet Spot")
    print("=" * 60)
    
    configurations = [
        {"name": "Tiny (Demo)", "vocab": 1000, "d_model": 128, "use_case": "Learning/prototyping"},
        {"name": "Small (Startup)", "vocab": 5000, "d_model": 256, "use_case": "Basic claims processing"},
        {"name": "Medium (nib)", "vocab": 15000, "d_model": 512, "use_case": "Production insurance"}, 
        {"name": "Large (Research)", "vocab": 50000, "d_model": 1024, "use_case": "Medical research AI"},
        {"name": "Huge (GPT-style)", "vocab": 100000, "d_model": 4096, "use_case": "General medical AI"}
    ]
    
    for config in configurations:
        print(f"\n🎯 {config['name']}")
        print(f"   vocab_size: {config['vocab']:,}")
        print(f"   d_model: {config['d_model']}")
        print(f"   Use case: {config['use_case']}")
        
        # Calculate capabilities
        total_patterns = config['vocab'] * config['d_model']
        print(f"   Total pattern capacity: {total_patterns:,}")
        
        # Memory estimation
        embedding_memory = (config['vocab'] * config['d_model'] * 4) / (1024 * 1024)
        print(f"   Embedding memory: {embedding_memory:.1f}MB")
        
        # Performance estimation  
        if config['vocab'] < 5000:
            coverage = "70% medical terms"
        elif config['vocab'] < 20000:
            coverage = "90% medical terms"
        else:
            coverage = "99% medical terms"
            
        if config['d_model'] < 256:
            understanding = "Basic"
        elif config['d_model'] < 512:
            understanding = "Good"  
        else:
            understanding = "Expert"
            
        print(f"   Expected coverage: {coverage}")
        print(f"   Understanding depth: {understanding}")

def medical_scaling_insights():
    """Specific insights for medical domain."""
    
    print("\n\n🏥 Medical Domain Insights")
    print("=" * 60)
    
    print("\n📊 Vocabulary Requirements by Medical Area:")
    areas = {
        "Basic insurance terms": 500,
        "Common medical conditions": 2000, 
        "All ICD-10 codes": 70000,
        "All drug names (FDA)": 20000,
        "Medical research terms": 100000,
        "Gene/protein names": 500000
    }
    
    for area, vocab_needed in areas.items():
        print(f"   {area:<25}: {vocab_needed:>8,} terms")
    
    print("\n🧠 d_model Requirements by Task:")
    tasks = {
        "Basic fraud detection": 128,
        "Medical relationship understanding": 256,
        "Complex diagnosis reasoning": 512, 
        "Drug interaction analysis": 1024,
        "Medical research comprehension": 2048
    }
    
    for task, d_model_needed in tasks.items():
        print(f"   {task:<30}: {d_model_needed:>4} dimensions")
        
    print("\n💡 Key Insights:")
    print("   • vocab_size: More = handle new/rare medical terms")
    print("   • d_model: More = deeper medical understanding") 
    print("   • Trade-off: Memory/speed vs capability")
    print("   • Sweet spot for nib: ~15K vocab, ~512 d_model")

if __name__ == "__main__":
    demonstrate_vocab_size_impact()
    demonstrate_d_model_impact() 
    demonstrate_combined_effect()
    medical_scaling_insights()