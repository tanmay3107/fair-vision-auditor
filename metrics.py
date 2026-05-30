# metrics.py
import numpy as np

class FairnessAuditor:
    def __init__(self, y_true, y_pred, sensitive_attribute, privileged_class=1):
        """
        Initializes the fairness evaluation engine.
        :param y_true: Ground truth labels (1 = Positive/Disease, 0 = Negative/Healthy).
        :param y_pred: Model's predicted labels.
        :param sensitive_attribute: Array indicating the group for each sample.
        :param privileged_class: The value in sensitive_attribute representing the majority/privileged group.
        """
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.Z = np.array(sensitive_attribute)
        self.priv_class = privileged_class
        
        # Create boolean masks to isolate the populations
        self.priv_mask = (self.Z == self.priv_class)
        self.unpriv_mask = (self.Z != self.priv_class)

    def _selection_rate(self, mask):
        """Calculates P(Y_hat = 1 | Z) - How often the model predicts positive for a group."""
        group_preds = self.y_pred[mask]
        if len(group_preds) == 0:
            return 0.0
        return np.mean(group_preds == 1)

    def _true_positive_rate(self, mask):
        """Calculates P(Y_hat = 1 | Y = 1, Z) - The Recall for a specific group."""
        # Isolate the actual positive cases within the specific group
        actual_positives = (self.y_true[mask] == 1)
        if np.sum(actual_positives) == 0:
            return 0.0
        
        # How many of those actual positives did the model correctly predict?
        correct_positives = (self.y_pred[mask][actual_positives] == 1)
        return np.mean(correct_positives)

    def demographic_parity_difference(self):
        """Difference in selection rates. Ideal value = 0.0"""
        sr_priv = self._selection_rate(self.priv_mask)
        sr_unpriv = self._selection_rate(self.unpriv_mask)
        return sr_unpriv - sr_priv

    def disparate_impact(self):
        """Ratio of selection rates. Ideal value = 1.0 (Safe range: 0.8 - 1.25)"""
        sr_priv = self._selection_rate(self.priv_mask)
        sr_unpriv = self._selection_rate(self.unpriv_mask)
        
        # Prevent division by zero
        if sr_priv == 0:
            return 0.0
        return sr_unpriv / sr_priv

    def equal_opportunity_difference(self):
        """Difference in True Positive Rates (Recall). Ideal value = 0.0"""
        tpr_priv = self._true_positive_rate(self.priv_mask)
        tpr_unpriv = self._true_positive_rate(self.unpriv_mask)
        return tpr_unpriv - tpr_priv

    def generate_audit_report(self):
        """Prints a structured fairness audit."""
        print("⚖️ FAIRNESS AUDIT REPORT ⚖️")
        print("-" * 30)
        
        dpd = self.demographic_parity_difference()
        di = self.disparate_impact()
        eod = self.equal_opportunity_difference()
        
        print(f"Demographic Parity Difference: {dpd:.4f} (Ideal: 0.0)")
        
        print(f"Disparate Impact Ratio:        {di:.4f} (Ideal: 1.0)")
        if 0.8 <= di <= 1.25:
            print("   ↳ Status: PASS (Within 80% rule)")
        else:
            print("   ↳ Status: FAIL (Evidence of bias)")
            
        print(f"Equal Opportunity Difference:  {eod:.4f} (Ideal: 0.0)")
        print("-" * 30)
        return {"DPD": dpd, "DI": di, "EOD": eod}


# --- Quick Test ---
if __name__ == "__main__":
    # Simulate a biased medical dataset (e.g., skin lesion classification)
    # y_true: 1 = Malignant, 0 = Benign
    # Z: 1 = Light Skin (Privileged in this dataset), 0 = Dark Skin (Unprivileged)
    
    y_true = [1, 1, 0, 0, 1, 1, 1, 0, 0, 1]
    
    # Notice the model misses the malignant cases (predicts 0) for the unprivileged group
    y_pred = [1, 1, 0, 0, 1, 0, 0, 0, 0, 0] 
    
    Z      = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0] 
    
    print("🔍 Auditing Baseline Model Predictions...")
    auditor = FairnessAuditor(y_true, y_pred, sensitive_attribute=Z, privileged_class=1)
    report = auditor.generate_audit_report()