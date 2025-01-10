import torch
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet
from PIL import Image
import os

def get_latest_model_path(base_dir: str):
    """Find the most recent trained model in the models directory."""
    models_dir = os.path.join(base_dir, 'models')
    if not os.path.exists(models_dir):
        return None
    
    # Get all run directories
    run_dirs = [d for d in os.listdir(models_dir) if d.startswith('run_')]
    if not run_dirs:
        return None
    
    # Get the latest run
    latest_run = sorted(run_dirs)[-1]
    run_dir = os.path.join(models_dir, latest_run)
    
    # Get the best model from this run
    model_files = [f for f in os.listdir(run_dir) if f.startswith('best_model_')]
    if not model_files:
        return None
    
    return os.path.join(models_dir, latest_run, model_files[0])

class DeepfakeDetector:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize the model with EfficientNet-B2
        model_version = 'efficientnet-b2'
        print(f"\nInitializing {model_version}")
        self.model = EfficientNet.from_pretrained(model_version, num_classes=2)
    
        # Get the base directory (where main.py is located)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Find the latest model
        model_path = get_latest_model_path(base_dir)
        if model_path is None:
            raise FileNotFoundError(
                "No trained model found! Please ensure:\n"
                "1. You have run the training script successfully\n"
                "2. The 'models' directory exists and contains trained models\n"
                "3. The model files follow the expected naming convention (best_model_*)"
            )
        
        try:
            print(f"Loading model from: {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print("Model loaded successfully!")
            
            # Print some model info if available
            if 'history' in checkpoint:
                best_val_acc = checkpoint['history'].get('best_val_acc', 0) * 100
                print(f"Model best validation accuracy: {best_val_acc:.2f}%")
                
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print("\nTroubleshooting steps:")
            print("1. Ensure you have run the training script successfully")
            print("2. Check if the model file is corrupted")
            print("3. Verify that you have sufficient memory")
            print(f"4. Confirm the model file exists at: {model_path}")
            raise

        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def predict(self, image_path):
        """
        Predict if a face image is real or fake.
        Returns probability of being fake (0-1 range).
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                output = self.model(image_tensor)
                probabilities = torch.softmax(output, dim=1)
                fake_prob = probabilities[0][1].item()
                
            return fake_prob
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return None
