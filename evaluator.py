# evaluator.py
import torch
import numpy as np
from metrics import FairnessAuditor

class SlicingEvaluator:
    def __init__(self, model, device='cpu'):
        """
        Initializes the evaluation pipeline.
        :param model: A trained PyTorch model (e.g., ResNet, MobileNet).
        :param device: Hardware accelerator ('cuda' or 'cpu').
        """
        self.model = model
        self.device = device
        self.model.to(self.device)
        
    def evaluate(self, dataloader, privileged_class=1):
        """
        Runs inference across the entire dataset and audits the results for bias.
        The dataloader MUST yield batches of (inputs, labels, sensitive_attributes).
        """
        self.model.eval()  # Lock layers like Dropout and BatchNorm
        
        all_y_true = []
        all_y_pred = []
        all_z = []
        
        print("🔍 Slicing Evaluator: Running batched inference...")
        
        # Disable gradient calculation to save massive amounts of VRAM
        with torch.no_grad():
            for batch_idx, (inputs, labels, sensitive_attrs) in enumerate(dataloader):
                inputs = inputs.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Convert raw logits to probabilities, then to binary predictions (Threshold = 0.5)
                probs = torch.sigmoid(outputs).squeeze()
                preds = (probs >= 0.5).long()
                
                # Move tensors back to CPU and convert to standard Python lists for aggregation
                all_y_true.extend(labels.cpu().tolist())
                all_y_pred.extend(preds.cpu().tolist())
                all_z.extend(sensitive_attrs.cpu().tolist())
                
        print("✅ Inference complete. Booting up Fairness Metrics Engine...\n")
        
        # Feed the aggregated arrays into our Day 1 module
        auditor = FairnessAuditor(
            y_true=all_y_true, 
            y_pred=all_y_pred, 
            sensitive_attribute=all_z, 
            privileged_class=privileged_class
        )
        
        # Run the audit and return the statistical dictionary
        return auditor.generate_audit_report()


# --- Quick Test ---
if __name__ == "__main__":
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    
    # 1. Simulate a very basic PyTorch Model
    class DummyClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 1)
            
        def forward(self, x):
            return self.fc(x)
            
    model = DummyClassifier()
    
    # 2. Simulate a PyTorch Dataset yielding (Image Features, Label, Sensitive Attribute)
    # We will simulate 100 patients. 
    # Let's say Z=1 is the privileged demographic, Z=0 is the unprivileged.
    num_samples = 100
    
    dummy_features = torch.randn(num_samples, 10)
    
    # Simulate ground truth (roughly 50% positive disease rate)
    dummy_labels = torch.randint(0, 2, (num_samples,)).float()
    
    # Simulate sensitive attributes (roughly 70% Privileged, 30% Unprivileged)
    dummy_sensitive = torch.where(torch.rand(num_samples) > 0.3, 1, 0)
    
    # 3. Create the specialized DataLoader
    dataset = TensorDataset(dummy_features, dummy_labels, dummy_sensitive)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    # 4. Execute the Audit Pipeline
    print("🧠 Booting FairVision-Auditor Pipeline...")
    evaluator = SlicingEvaluator(model, device='cpu')
    audit_results = evaluator.evaluate(dataloader, privileged_class=1)