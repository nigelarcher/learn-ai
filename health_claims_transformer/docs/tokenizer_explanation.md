# Medical Terminology Tokenizer Explained

## What This File Does

The `tokenizer.py` file implements a specialized tokenizer for medical and insurance terminology. It handles:
- Medical codes (ICD-10, CPT, HCPCS)
- Drug names and dosages
- Insurance policy terms
- Monetary values
- Clinical abbreviations
- Subword tokenization for rare medical terms

## Why It's Important

For health insurance claims:

- **Code Preservation**: Medical codes like "E11.9" (Type 2 diabetes) must stay intact
- **Terminology Handling**: Complex drug names need consistent tokenization
- **Numeric Processing**: Dosages (5mg) and costs ($1,234.56) require special handling
- **Abbreviation Understanding**: Medical abbreviations (PRN, BID, QD) are domain-specific

## How It Contributes to Learning

Building a medical tokenizer teaches:

1. **Domain Adaptation**: How general NLP techniques need modification for specialized domains
2. **Tokenization Trade-offs**: Vocabulary size vs. sequence length
3. **Subword Algorithms**: BPE-like approaches for rare medical terms
4. **Preprocessing Impact**: How tokenization affects model performance

## Tokenization Strategy

### Medical Code Detection
```
ICD-10: [A-Z]\d{2}\.?\d*
CPT: \d{5}
HCPCS: [A-Z]\d{4}
```

### Drug Name Processing
- Preserve brand names (Lipitor, Metformin)
- Handle dosages (500mg, 10mL)
- Recognize formulations (extended-release, oral)

### Monetary Value Handling
- Normalize currency symbols
- Preserve decimal precision
- Handle ranges ($100-$500)

## Vocabulary Composition

Typical medical claims vocabulary:
- 30% Medical codes
- 25% Common medical terms
- 20% Drug names
- 15% Insurance terms
- 10% General language

## Challenges Addressed

1. **Rare Diseases**: Handle conditions that appear once in millions of claims
2. **New Drugs**: Accommodate newly approved medications
3. **Typos**: Medical records often contain misspellings
4. **Multi-lingual**: Patient names and some terms may be non-English