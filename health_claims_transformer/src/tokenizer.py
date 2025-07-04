import re
import numpy as np
from collections import Counter, defaultdict


class MedicalTokenizer:
    """Tokenizer specialized for medical and insurance claims text."""
    
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.word_to_id = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3, "<NUM>": 4}
        self.id_to_word = {v: k for k, v in self.word_to_id.items()}
        self.next_id = 5
        
        self.medical_code_patterns = {
            'icd10': re.compile(r'\b[A-Z]\d{2}\.?\d*\b'),
            'cpt': re.compile(r'\b\d{5}\b'),
            'hcpcs': re.compile(r'\b[A-Z]\d{4}\b'),
            'ndc': re.compile(r'\b\d{4,5}-\d{3,4}-\d{2}\b'),
        }
        
        self.medical_abbreviations = {
            'PRN', 'BID', 'TID', 'QID', 'QD', 'PO', 'IV', 'IM', 'SC',
            'MG', 'ML', 'MCG', 'IU', 'STAT', 'NPO', 'WNL', 'NAD'
        }
        
        self.drug_pattern = re.compile(r'\b\w+(?:azole|cillin|pril|sartan|statin|olol|azepam)\b', re.I)
        self.dosage_pattern = re.compile(r'\b\d+\.?\d*\s*(?:mg|ml|mcg|g|kg|IU|units?)\b', re.I)
        self.money_pattern = re.compile(r'\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})?')
        
    def _extract_medical_codes(self, text):
        """Extract and preserve medical codes."""
        codes = []
        for code_type, pattern in self.medical_code_patterns.items():
            matches = pattern.findall(text)
            for match in matches:
                codes.append((match, f'<{code_type.upper()}>'))
                text = text.replace(match, f' <{code_type.upper()}> ')
        return text, codes
    
    def _normalize_numbers(self, text):
        """Normalize numeric values while preserving medical context."""
        dosages = self.dosage_pattern.findall(text)
        for dosage in dosages:
            text = text.replace(dosage, f' <DOSAGE> ')
        
        money_amounts = self.money_pattern.findall(text)
        for amount in money_amounts:
            text = text.replace(amount, ' <MONEY> ')
        
        remaining_numbers = re.findall(r'\b\d+\.?\d*\b', text)
        for num in remaining_numbers:
            text = text.replace(num, ' <NUM> ')
        
        return text
    
    def _tokenize_medical_terms(self, text):
        """Tokenize with special handling for medical terms."""
        text = text.upper()
        
        for abbrev in self.medical_abbreviations:
            text = re.sub(f'\\b{abbrev}\\b', f' {abbrev} ', text)
        
        text, codes = self._extract_medical_codes(text)
        
        text = self._normalize_numbers(text)
        
        tokens = []
        words = text.split()
        
        for word in words:
            word = word.strip('.,;:!?\'"')
            
            if not word:
                continue
            
            if word.startswith('<') and word.endswith('>'):
                tokens.append(word)
            elif self.drug_pattern.match(word):
                tokens.append('<DRUG>')
                subwords = self._subword_tokenize(word)
                tokens.extend(subwords)
            else:
                tokens.append(word)
        
        return tokens
    
    def _subword_tokenize(self, word):
        """Simple subword tokenization for OOV medical terms."""
        if len(word) <= 3:
            return [word]
        
        subwords = []
        i = 0
        while i < len(word):
            found = False
            for j in range(len(word), i, -1):
                subword = word[i:j]
                if subword in self.word_to_id or len(subword) <= 4:
                    subwords.append(subword)
                    i = j
                    found = True
                    break
            
            if not found:
                subwords.append(word[i])
                i += 1
        
        return subwords
    
    def build_vocabulary(self, texts):
        """Build vocabulary from training texts."""
        word_counts = Counter()
        
        for text in texts:
            tokens = self._tokenize_medical_terms(text)
            word_counts.update(tokens)
        
        most_common = word_counts.most_common(self.vocab_size - len(self.word_to_id))
        
        for word, _ in most_common:
            if word not in self.word_to_id:
                self.word_to_id[word] = self.next_id
                self.id_to_word[self.next_id] = word
                self.next_id += 1
    
    def encode(self, text, max_length=None):
        """Encode text to token IDs."""
        tokens = self._tokenize_medical_terms(text)
        
        ids = []
        for token in tokens:
            if token in self.word_to_id:
                ids.append(self.word_to_id[token])
            else:
                subwords = self._subword_tokenize(token)
                for subword in subwords:
                    ids.append(self.word_to_id.get(subword, self.word_to_id['<UNK>']))
        
        ids = [self.word_to_id['<BOS>']] + ids + [self.word_to_id['<EOS>']]
        
        if max_length:
            if len(ids) > max_length:
                ids = ids[:max_length-1] + [self.word_to_id['<EOS>']]
            else:
                ids = ids + [self.word_to_id['<PAD>']] * (max_length - len(ids))
        
        return np.array(ids)
    
    def decode(self, ids):
        """Decode token IDs back to text."""
        tokens = []
        for id in ids:
            if id in self.id_to_word:
                token = self.id_to_word[id]
                if token not in ['<PAD>', '<BOS>', '<EOS>']:
                    tokens.append(token)
        
        text = ' '.join(tokens)
        
        text = re.sub(r'\s+([.,;:!?\'""])', r'\1', text)
        
        return text
    
    def batch_encode(self, texts, max_length=None):
        """Encode multiple texts with padding."""
        if max_length is None:
            max_length = max(len(self._tokenize_medical_terms(text)) + 2 for text in texts)
        
        batch = []
        for text in texts:
            encoded = self.encode(text, max_length)
            batch.append(encoded)
        
        return np.array(batch)


class ClaimsVocabulary:
    """Specialized vocabulary for insurance claims processing."""
    
    def __init__(self):
        self.diagnosis_codes = {}
        self.procedure_codes = {}
        self.drug_codes = {}
        self.policy_terms = set()
        
        self._load_medical_codes()
        self._load_policy_vocabulary()
    
    def _load_medical_codes(self):
        """Load common medical codes (would load from file in practice)."""
        self.diagnosis_codes = {
            'E11.9': 'Type 2 diabetes without complications',
            'I10': 'Essential hypertension',
            'J45.909': 'Unspecified asthma, uncomplicated',
            'M79.3': 'Myalgia',
            'F41.9': 'Anxiety disorder, unspecified'
        }
        
        self.procedure_codes = {
            '99213': 'Office visit, established patient, low complexity',
            '99214': 'Office visit, established patient, moderate complexity',
            '36415': 'Routine venipuncture',
            '80053': 'Comprehensive metabolic panel',
            '81001': 'Urinalysis, automated'
        }
    
    def _load_policy_vocabulary(self):
        """Load insurance policy terms."""
        self.policy_terms = {
            'deductible', 'copay', 'coinsurance', 'out-of-pocket',
            'in-network', 'out-of-network', 'prior-authorization',
            'formulary', 'tier', 'coverage', 'exclusion', 'limitation',
            'benefit', 'claim', 'appeal', 'grievance', 'provider'
        }
    
    def get_code_description(self, code):
        """Get description for a medical code."""
        if code in self.diagnosis_codes:
            return self.diagnosis_codes[code]
        elif code in self.procedure_codes:
            return self.procedure_codes[code]
        else:
            return None