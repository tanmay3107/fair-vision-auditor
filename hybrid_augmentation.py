# hybrid_augmentation.py
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import random

class FairnessAwareDataset(Dataset):
    def __init__(self, base_dataset, privileged_class=1):
        """
        Wraps a standard PyTorch dataset to apply Hybrid Augmentation
        specifically targeting unprivileged edge cases.
        
        :param base_dataset: A dataset returning (image_tensor, label, sensitive_attribute)
        :param privileged_class: The integer representing the majority/easy demographic
        """
        self.base_dataset = base_dataset
        self.priv_class = privileged_class
        
        # Standard augmentations for the privileged/majority class
        self.standard_transforms = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(10)
        ])
        
        # Aggressive "Hybrid" augmentations for the unprivileged class
        # This forces the network to stop relying on lighting, contrast, or skin-tone artifacts
        self.hybrid_transforms = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(30),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            T.RandomErasing(p=0.3, scale=(0.02, 0.1)) # Forces the model to look at the whole image
        ])

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        # Extract the raw data from the underlying dataset
        image, label, sensitive_attr = self.base_dataset[idx]
        
        # Dynamically route the augmentation based on the sensitive attribute
        if sensitive_attr.item() == self.priv_class:
            # Apply standard, gentle transformations
            augmented_image = self.standard_transforms(image)
        else:
            # Apply aggressive hybrid transformations to robustify the minority features
            augmented_image = self.hybrid_transforms(image)
            
        return augmented_image, label, sensitive_attr


# --- Quick Test ---
if __name__ == "__main__":
    from torch.utils.data import TensorDataset
    
    # Simulate a medical image tensor: 3 channels, 224x224 pixels
    dummy_images = torch.randn(10, 3, 224, 224)
    dummy_labels = torch.randint(0, 2, (10,))
    
    # Simulate sensitive attributes (e.g., 1 = Standard lighting, 0 = Poor lighting)
    dummy_sensitive = torch.tensor([1, 1, 0, 1, 0, 0, 1, 1, 0, 1])
    
    # Create the base dataset
    base_data = TensorDataset(dummy_images, dummy_labels, dummy_sensitive)
    
    print("🧠 Initializing Hybrid Augmentation Pipeline...")
    fair_dataset = FairnessAwareDataset(base_data, privileged_class=1)
    
    # Test the pipeline execution
    print("\n🔄 Simulating DataLoader extraction:")
    for i in range(3):
        img, lbl, z = fair_dataset[i]
        group = "Privileged" if z.item() == 1 else "Unprivileged"
        print(f"Sample {i}: Group = {group} | Output Tensor Shape: {img.shape}")
        
    print("\n✅ Pipeline routing successfully isolated and augmented the unprivileged slices.")