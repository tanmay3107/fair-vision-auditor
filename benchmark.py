# benchmark.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Importing our custom modules from the previous days
from evaluator import SlicingEvaluator
from hybrid_augmentation import FairnessAwareDataset

def train_model(model, dataloader, epochs=5, lr=0.001):
    """A standard PyTorch training loop."""
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        for inputs, labels, _ in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    return model

# --- Quick Test & Benchmark Execution ---
if __name__ == "__main__":
    # 1. Define a lightweight CNN architecture for the benchmark
    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
            )
            self.classifier = nn.Linear(32 * 56 * 56, 1)

        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            return self.classifier(x)

    print("📊 Generating Simulated Imbalanced Medical Dataset...")
    # Simulate 200 patients (Images: 3 channels, 224x224)
    N = 200
    images = torch.randn(N, 3, 224, 224)
    labels = torch.randint(0, 2, (N,)).float()
    
    # 80% Privileged (Easy lighting/conditions), 20% Unprivileged (Hard conditions)
    sensitive_attrs = torch.where(torch.rand(N) > 0.2, 1, 0)
    
    # Create the standard dataset (Baseline)
    base_dataset = TensorDataset(images, labels, sensitive_attrs)
    baseline_loader = DataLoader(base_dataset, batch_size=16, shuffle=True)
    
    # Create the Hybrid dataset (Mitigation)
    hybrid_dataset = FairnessAwareDataset(base_dataset, privileged_class=1)
    hybrid_loader = DataLoader(hybrid_dataset, batch_size=16, shuffle=True)
    
    # We also need a separate test loader for evaluation (no shuffling, no augmentation)
    test_loader = DataLoader(base_dataset, batch_size=16, shuffle=False)

    # 2. Train and Evaluate the Baseline Model
    print("\n🧠 Training BASELINE Model (Standard Data Pipeline)...")
    baseline_model = SimpleCNN()
    baseline_model = train_model(baseline_model, baseline_loader, epochs=3)
    
    print("🔍 Auditing Baseline Model:")
    evaluator_base = SlicingEvaluator(baseline_model, device='cpu')
    base_metrics = evaluator_base.evaluate(test_loader)

    # 3. Train and Evaluate the Hybrid Model
    print("\n🛡️ Training HYBRID Model (Fairness-Aware Data Pipeline)...")
    hybrid_model = SimpleCNN()
    hybrid_model = train_model(hybrid_model, hybrid_loader, epochs=3)
    
    print("🔍 Auditing Hybrid Model:")
    evaluator_hybrid = SlicingEvaluator(hybrid_model, device='cpu')
    hybrid_metrics = evaluator_hybrid.evaluate(test_loader)

    # 4. The Final Comparative Analysis
    print("\n🏆 BENCHMARK RESULTS: COMPARATIVE FAIRNESS DELTA 🏆")
    print("-" * 50)
    print(f"{'Metric':<30} | {'Baseline':<10} | {'Hybrid (Ours)':<10}")
    print("-" * 50)
    
    for metric in ["DPD", "DI", "EOD"]:
        b_val = base_metrics[metric]
        h_val = hybrid_metrics[metric]
        
        # We want DPD and EOD close to 0.0, and DI close to 1.0
        if metric in ["DPD", "EOD"]:
            improvement = abs(b_val) - abs(h_val)
        else: # DI
            improvement = abs(1.0 - b_val) - abs(1.0 - h_val)
            
        trend = "✅ Improved" if improvement > 0 else "❌ Regressed"
        print(f"{metric:<30} | {b_val:>8.4f}   | {h_val:>8.4f}      ({trend})")