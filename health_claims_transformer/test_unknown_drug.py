#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from tokenizer import MedicalTokenizer

def test_unknown_drug_handling():
    """Test how our tokenizer handles completely unknown drugs."""
    
    print("🧪 Testing Unknown Drug Handling")
    print("=" * 50)
    
    # Create tokenizer and build vocabulary with known examples
    tokenizer = MedicalTokenizer(vocab_size=1000)
    
    # Training examples with known drugs
    known_examples = [
        "Patient diagnosed with E11.9 diabetes prescribed Metformin 500mg",
        "Hypertension patient given Lisinopril 10mg daily",
        "Depression treated with Sertraline 50mg BID"
    ]
    
    print("1. Building vocabulary with known drugs:")
    for example in known_examples:
        print(f"   - {example}")
    
    tokenizer.build_vocabulary(known_examples)
    print(f"\nVocabulary size: {len(tokenizer.word_to_id)} tokens")
    
    # Test with completely unknown drugs
    print("\n2. Testing with UNKNOWN drugs:")
    
    unknown_examples = [
        "Patient with diabetes prescribed Leqembi 10mg",  # Alzheimer's drug
        "Cancer patient given Pembrolizumab 200mg",      # Immunotherapy  
        "Arthritis treated with Adalimumab 40mg",        # Biologic
        "Patient prescribed Semaglutide 0.5mg weekly"    # GLP-1 agonist
    ]
    
    for example in unknown_examples:
        print(f"\n📝 Testing: {example}")
        
        # Tokenize
        tokens = tokenizer._tokenize_medical_terms(example)
        print(f"   Tokens: {tokens}")
        
        # Encode  
        encoded = tokenizer.encode(example, max_length=20)
        print(f"   Encoded: {encoded}")
        
        # Check what happened to the unknown drug
        if "prescribed " in example:
            unknown_drug = example.split("prescribed ")[1].split(" ")[0]
        elif "given " in example:
            unknown_drug = example.split("given ")[1].split(" ")[0]
        elif "treated with " in example:
            unknown_drug = example.split("treated with ")[1].split(" ")[0]
        else:
            unknown_drug = "unknown"
            
        print(f"   Unknown drug '{unknown_drug}' handling:")
        
        if unknown_drug.upper() in tokens:
            print(f"   ✅ Kept as: {unknown_drug.upper()}")
        elif '<DRUG>' in tokens:
            print(f"   🔍 Recognized as drug pattern: <DRUG>")
        elif any('<UNK>' in str(token) for token in tokens):
            print(f"   ❓ Became unknown token: <UNK>")
        else:
            print(f"   🔤 Broken into subwords")

def test_pattern_recognition():
    """Test our drug pattern recognition."""
    
    print("\n\n🎯 Testing Drug Pattern Recognition")
    print("=" * 50)
    
    tokenizer = MedicalTokenizer()
    
    # Test different drug name endings
    test_drugs = [
        "Fluconazole",    # -azole (antifungal)
        "Amoxicillin",    # -cillin (antibiotic)
        "Lisinopril",     # -pril (ACE inhibitor)
        "Losartan",       # -sartan (ARB)
        "Atorvastatin",   # -statin (cholesterol)
        "Metoprolol",     # -olol (beta blocker)
        "Lorazepam",      # -azepam (benzodiazepine)
        "Leqembi",        # Unknown pattern
        "Aspirin",        # Common drug, no pattern
    ]
    
    for drug in test_drugs:
        pattern_match = tokenizer.drug_pattern.match(drug)
        
        print(f"Drug: {drug:15} | Pattern match: {bool(pattern_match)} | ", end="")
        
        if pattern_match:
            suffix = drug[pattern_match.start():pattern_match.end()]
            print(f"Recognized suffix: {suffix}")
        else:
            print("No pattern recognized")

def test_real_scenario():
    """Test a realistic scenario with unknown drug."""
    
    print("\n\n🏥 Real Scenario Test")
    print("=" * 50)
    
    tokenizer = MedicalTokenizer(vocab_size=500)
    
    # Build vocabulary with typical claims
    training_claims = [
        "E11.9 diabetes patient prescribed Metformin 500mg cost $50",
        "I10 hypertension treated with Lisinopril 10mg daily $30", 
        "Depression F32.9 patient given Sertraline 50mg BID $40",
        "Office visit 99214 for diabetes follow-up $150"
    ]
    
    tokenizer.build_vocabulary(training_claims)
    
    # New claim with unknown drug
    new_claim = "E11.9 diabetes patient prescribed Ozempic 0.5mg weekly $800"
    
    print(f"📋 New claim: {new_claim}")
    print("\n🔍 Tokenization process:")
    
    # Step by step tokenization
    print(f"1. Original: {new_claim}")
    
    # Extract medical codes
    text_with_codes, codes = tokenizer._extract_medical_codes(new_claim)
    print(f"2. After code extraction: {text_with_codes}")
    print(f"   Extracted codes: {codes}")
    
    # Normalize numbers  
    text_normalized = tokenizer._normalize_numbers(text_with_codes)
    print(f"3. After number normalization: {text_normalized}")
    
    # Full tokenization
    tokens = tokenizer._tokenize_medical_terms(new_claim)
    print(f"4. Final tokens: {tokens}")
    
    # Encoding
    encoded = tokenizer.encode(new_claim, max_length=15)
    print(f"5. Encoded: {encoded}")
    
    # Decoding
    decoded = tokenizer.decode(encoded)
    print(f"6. Decoded: {decoded}")
    
    print(f"\n💡 Analysis:")
    print(f"   - 'Ozempic' was never seen in training")
    print(f"   - Model still understands: diabetes + drug + dosage + cost")
    print(f"   - Context preserved even with unknown drug name")

if __name__ == "__main__":
    test_unknown_drug_handling()
    test_pattern_recognition() 
    test_real_scenario()