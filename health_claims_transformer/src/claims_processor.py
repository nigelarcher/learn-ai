import numpy as np
from tokenizer import MedicalTokenizer, ClaimsVocabulary
from transformer import Transformer


class ClaimsProcessor:
    """Process insurance claims using transformer model."""
    
    def __init__(self, model, tokenizer, vocab):
        """
        Args:
            model: Trained transformer model
            tokenizer: Medical tokenizer
            vocab: Claims vocabulary
        """
        self.model = model
        self.tokenizer = tokenizer
        self.vocab = vocab
        
        self.cost_statistics = self._load_cost_statistics()
        self.fraud_patterns = self._load_fraud_patterns()
        
    def _load_cost_statistics(self):
        """Load statistical data for anomaly detection."""
        return {
            '99213': {'mean': 125.50, 'std': 25.30},
            '99214': {'mean': 185.75, 'std': 35.20},
            '80053': {'mean': 45.00, 'std': 10.50},
            '36415': {'mean': 15.25, 'std': 5.00},
        }
    
    def _load_fraud_patterns(self):
        """Load known fraud patterns."""
        return {
            'duplicate_billing': {
                'time_window': 30,
                'similarity_threshold': 0.95
            },
            'upcoding': {
                'common_pairs': [('99213', '99214'), ('99214', '99215')]
            },
            'unusual_volume': {
                'daily_limit': 50,
                'procedure_limit': 20
            }
        }
    
    def process_claim(self, claim_text, claim_metadata):
        """
        Process a single insurance claim.
        
        Args:
            claim_text: Raw claim text
            claim_metadata: Dictionary with date, provider, amounts, etc.
        
        Returns:
            Dictionary with assessment results
        """
        features = self._extract_features(claim_text, claim_metadata)
        
        encoded = self.tokenizer.encode(claim_text)
        
        model_output = self._run_inference(encoded)
        
        assessment = self._assess_claim(features, model_output)
        
        anomalies = self._detect_anomalies(features, model_output, claim_metadata)
        
        explanation = self._generate_explanation(assessment, anomalies)
        
        return {
            'approved': assessment['approved'],
            'payment_amount': assessment['payment_amount'],
            'confidence': assessment['confidence'],
            'anomalies': anomalies,
            'explanation': explanation,
            'attention_weights': model_output.get('attention_weights')
        }
    
    def _extract_features(self, text, metadata):
        """Extract structured features from claim."""
        features = {
            'diagnosis_codes': [],
            'procedure_codes': [],
            'total_charges': metadata.get('total_charges', 0),
            'provider_id': metadata.get('provider_id'),
            'patient_age': metadata.get('patient_age'),
            'claim_date': metadata.get('claim_date')
        }
        
        text_upper = text.upper()
        
        for pattern_name, pattern in self.tokenizer.medical_code_patterns.items():
            matches = pattern.findall(text_upper)
            if pattern_name == 'icd10':
                features['diagnosis_codes'].extend(matches)
            elif pattern_name == 'cpt':
                features['procedure_codes'].extend(matches)
        
        return features
    
    def _run_inference(self, encoded_claim):
        """Run claim through transformer model."""
        batch = encoded_claim.reshape(1, -1)
        
        dummy_target = np.zeros_like(batch)
        
        logits = self.model.forward(batch, dummy_target, training=False)
        
        probs = self._softmax(logits[0])
        
        confidence = np.max(probs, axis=-1).mean()
        
        return {
            'logits': logits,
            'probabilities': probs,
            'confidence': confidence,
            'attention_weights': self._extract_attention_patterns()
        }
    
    def _softmax(self, x):
        """Compute softmax probabilities."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def _extract_attention_patterns(self):
        """Extract attention patterns from model (simplified)."""
        return None
    
    def _assess_claim(self, features, model_output):
        """Assess claim validity and calculate payment."""
        confidence = model_output['confidence']
        
        total_allowed = 0
        for proc_code in features['procedure_codes']:
            if proc_code in self.cost_statistics:
                stats = self.cost_statistics[proc_code]
                total_allowed += stats['mean']
        
        if confidence < 0.7:
            approved = False
            payment = 0
        elif features['total_charges'] > total_allowed * 2:
            approved = False
            payment = 0
        else:
            approved = True
            payment = min(features['total_charges'], total_allowed)
        
        return {
            'approved': approved,
            'payment_amount': payment,
            'confidence': confidence,
            'allowed_amount': total_allowed
        }
    
    def _detect_anomalies(self, features, model_output, metadata):
        """Detect potential fraud or errors."""
        anomalies = []
        
        for proc_code in features['procedure_codes']:
            if proc_code in self.cost_statistics:
                stats = self.cost_statistics[proc_code]
                proc_charges = self._get_procedure_charge(proc_code, metadata)
                
                if proc_charges:
                    z_score = abs(proc_charges - stats['mean']) / stats['std']
                    if z_score > 3:
                        anomalies.append({
                            'type': 'unusual_cost',
                            'procedure': proc_code,
                            'severity': 'high',
                            'details': f'Cost ${proc_charges:.2f} is {z_score:.1f} std devs from mean'
                        })
        
        diag_proc_pairs = [
            (d, p) for d in features['diagnosis_codes'] 
            for p in features['procedure_codes']
        ]
        
        unusual_pairs = self._check_unusual_combinations(diag_proc_pairs)
        anomalies.extend(unusual_pairs)
        
        if model_output['confidence'] < 0.5:
            anomalies.append({
                'type': 'low_confidence',
                'severity': 'medium',
                'details': f'Model confidence {model_output["confidence"]:.2%} below threshold'
            })
        
        provider_anomalies = self._check_provider_patterns(metadata.get('provider_id'))
        anomalies.extend(provider_anomalies)
        
        return anomalies
    
    def _get_procedure_charge(self, proc_code, metadata):
        """Extract charge for specific procedure."""
        return metadata.get('total_charges', 0) / len(metadata.get('procedures', [1]))
    
    def _check_unusual_combinations(self, pairs):
        """Check for unusual diagnosis-procedure combinations."""
        anomalies = []
        
        unusual = [
            ('E11.9', '99215'),
            ('I10', '36415'),
        ]
        
        for pair in pairs:
            if pair not in unusual:
                anomalies.append({
                    'type': 'unusual_combination',
                    'severity': 'low',
                    'details': f'Rare combination: {pair[0]} with {pair[1]}'
                })
        
        return anomalies
    
    def _check_provider_patterns(self, provider_id):
        """Check for unusual provider billing patterns."""
        return []
    
    def _generate_explanation(self, assessment, anomalies):
        """Generate human-readable explanation."""
        explanation = []
        
        if assessment['approved']:
            explanation.append(f"Claim approved for ${assessment['payment_amount']:.2f}")
            explanation.append(f"Allowed amount: ${assessment['allowed_amount']:.2f}")
        else:
            explanation.append("Claim denied")
            
            if assessment['confidence'] < 0.7:
                explanation.append("Reason: Low confidence in claim validity")
            elif assessment['payment_amount'] == 0:
                explanation.append("Reason: Charges exceed reasonable limits")
        
        if anomalies:
            explanation.append(f"\nDetected {len(anomalies)} anomalies:")
            for anomaly in anomalies[:3]:
                explanation.append(f"- {anomaly['type']}: {anomaly['details']}")
        
        explanation.append(f"\nConfidence: {assessment['confidence']:.1%}")
        
        return '\n'.join(explanation)


class AnomalyDetector:
    """Advanced anomaly detection for claims."""
    
    def __init__(self, historical_data=None):
        self.historical_data = historical_data or {}
        self.thresholds = self._initialize_thresholds()
        
    def _initialize_thresholds(self):
        """Initialize detection thresholds."""
        return {
            'duplicate_similarity': 0.95,
            'temporal_window': 30,
            'volume_daily': 50,
            'cost_zscore': 3.0,
            'rare_combo_threshold': 0.001
        }
    
    def detect_duplicate_claims(self, claim, recent_claims):
        """Detect potential duplicate submissions."""
        duplicates = []
        
        for recent in recent_claims:
            similarity = self._calculate_similarity(claim, recent)
            
            if similarity > self.thresholds['duplicate_similarity']:
                days_apart = abs((claim['date'] - recent['date']).days)
                
                if days_apart < self.thresholds['temporal_window']:
                    duplicates.append({
                        'type': 'duplicate_claim',
                        'severity': 'high',
                        'match_id': recent['id'],
                        'similarity': similarity,
                        'days_apart': days_apart
                    })
        
        return duplicates
    
    def _calculate_similarity(self, claim1, claim2):
        """Calculate similarity between two claims."""
        if claim1.get('provider_id') != claim2.get('provider_id'):
            return 0.0
        
        proc_match = len(set(claim1['procedures']) & set(claim2['procedures']))
        proc_total = len(set(claim1['procedures']) | set(claim2['procedures']))
        
        if proc_total == 0:
            return 0.0
        
        return proc_match / proc_total
    
    def detect_billing_patterns(self, provider_claims):
        """Detect unusual billing patterns by provider."""
        anomalies = []
        
        daily_counts = defaultdict(int)
        for claim in provider_claims:
            daily_counts[claim['date']] += 1
        
        for date, count in daily_counts.items():
            if count > self.thresholds['volume_daily']:
                anomalies.append({
                    'type': 'high_volume',
                    'severity': 'medium',
                    'date': date,
                    'count': count,
                    'threshold': self.thresholds['volume_daily']
                })
        
        return anomalies