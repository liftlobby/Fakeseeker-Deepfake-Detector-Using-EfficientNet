# Core ML libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from efficientnet_pytorch import EfficientNet

# Data processing
from PIL import Image
import os
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
from datetime import datetime
import cv2
from facenet_pytorch import MTCNN
import tempfile
from typing import List, Tuple

# Visualization and metrics
import shutil
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

class FaceExtractor:
    def __init__(self, device='cpu'):
        self.device = device
        self.mtcnn = MTCNN(margin=14, keep_all=True, factor=0.7, device=device)
        self.temp_dir = tempfile.mkdtemp()

    def extract_faces_from_image(self, image_path: str) -> List[Image.Image]:
        """Extract faces from a single image file."""
        try:
            img = Image.open(image_path).convert('RGB')
            faces = self.mtcnn(img)
            if faces is None:
                return []
                
            boxes = self.mtcnn.detect(img)[0]
            if boxes is None:
                return []

            extracted_faces = []
            for box in boxes:
                box = [int(b) for b in box]
                face = img.crop((box[0], box[1], box[2], box[3]))
                extracted_faces.append(face)
            
            return extracted_faces
        except Exception as e:
            print(f"Error extracting faces from image: {str(e)}")
            return []

    def extract_faces_from_video(self, video_path: str, n_frames: int = 20) -> List[Image.Image]:
        """Extract faces from video frames."""
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Calculate frame indices to extract
            if n_frames > total_frames:
                n_frames = total_frames
            frame_indices = np.linspace(0, total_frames-1, n_frames, dtype=int)
            
            extracted_faces = []
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                
                # Detect faces
                faces = self.mtcnn(frame_pil)
                if faces is None:
                    continue
                    
                boxes = self.mtcnn.detect(frame_pil)[0]
                if boxes is None:
                    continue
                
                # Extract faces
                for box in boxes:
                    box = [int(b) for b in box]
                    face = frame_pil.crop((box[0], box[1], box[2], box[3]))
                    extracted_faces.append(face)
            
            cap.release()
            return extracted_faces
        except Exception as e:
            print(f"Error extracting faces from video: {str(e)}")
            return []

    def process_dataset(self, data_dir: str, output_dir: str) -> Tuple[List[str], List[int]]:
        """Process entire dataset and extract faces."""
        os.makedirs(output_dir, exist_ok=True)
        processed_paths = []
        labels = []
        
        # Process real and fake directories
        for label, subdir in enumerate(['real', 'fake']):
            input_subdir = os.path.join(data_dir, subdir)
            output_subdir = os.path.join(output_dir, subdir)
            os.makedirs(output_subdir, exist_ok=True)
            
            files = [f for f in os.listdir(input_subdir) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.mp4', '.avi', '.mov'))]
            
            for file in tqdm(files, desc=f'Processing {subdir} files'):
                input_path = os.path.join(input_subdir, file)
                
                # Extract faces based on file type
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    faces = self.extract_faces_from_image(input_path)
                else:  # video file
                    faces = self.extract_faces_from_video(input_path)
                
                # Save extracted faces
                for i, face in enumerate(faces):
                    output_filename = f"{os.path.splitext(file)[0]}_face_{i}.jpg"
                    output_path = os.path.join(output_subdir, output_filename)
                    face.save(output_path)
                    processed_paths.append(output_path)
                    labels.append(label)
        
        return processed_paths, labels

    def cleanup(self):
        """Clean up temporary directory."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

class DeepfakeDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return image, label
        except Exception as e:
            print(f"Error loading image {image_path}: {str(e)}")
            return torch.zeros((3, 224, 224)), torch.tensor(0)

class DeepfakeTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name()}")
        
        # Create timestamp for this training run
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = os.path.join(self.config['save_dir'], f'run_{self.timestamp}')
        os.makedirs(self.run_dir, exist_ok=True)
        
        # Create model
        self.model = EfficientNet.from_pretrained(self.config['model_version'], num_classes=2)
        self.model = self.model.to(self.device)
        
        # Define loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.1, patience=5, verbose=True
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'best_val_acc': 0.0
        }
        
        # Setup data transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])

    def prepare_data(self):
        """Prepare dataset with enhanced balancing techniques."""
        # Extract faces and get paths/labels
        face_extractor = FaceExtractor(self.device)
        processed_dir = os.path.join(self.config['data_dir'], 'processed_faces')
        
        if os.path.exists(processed_dir):
            print(f"\nRemoving old processed faces from {processed_dir}")
            shutil.rmtree(processed_dir)
        
        print("\nExtracting faces from dataset...")
        image_paths, labels = face_extractor.process_dataset(
            self.config['data_dir'],
            processed_dir
        )
        face_extractor.cleanup()
        
        if not image_paths:
            raise ValueError("No faces were extracted from the dataset!")
        
        # Analyze class distribution
        real_count = sum(1 for label in labels if label == 0)
        fake_count = sum(1 for label in labels if label == 1)
        print(f"\nInitial class distribution:")
        print(f"Real faces: {real_count} ({real_count/(real_count+fake_count):.2%})")
        print(f"Fake faces: {fake_count} ({fake_count/(real_count+fake_count):.2%})")
        
        # Setup weighted loss
        self.setup_weighted_loss(labels)
        
        # Split dataset with stratification
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            image_paths, labels,
            test_size=self.config.get('val_split', 0.2),
            random_state=42,
            stratify=labels
        )
        
        # Determine minority class and create datasets
        minority_class = 0 if real_count < fake_count else 1
        train_dataset = DeepfakeDataset(
            train_paths,
            train_labels,
            transform=self.get_augmented_transforms(
                is_minority_class=(np.array(train_labels) == minority_class).any()
            )
        )
        
        val_dataset = DeepfakeDataset(
            val_paths,
            val_labels,
            transform=self.get_augmented_transforms(is_minority_class=False)
        )
        
        # Create balanced dataloaders
        self.train_loader = self.create_balanced_dataloader(train_dataset, train_labels)
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True
        )
        
        # Monitor final class distribution
        print("\nMonitoring class distribution after balancing:")
        train_dist = self.monitor_class_distribution(self.train_loader, 'train')
        val_dist = self.monitor_class_distribution(self.val_loader, 'validation')
        
        print(f"\nDataset prepared:")
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")

    # Training
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(self.train_loader, desc='Training')
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            progress_bar.set_postfix({
                'loss': total_loss / (progress_bar.n + 1),
                'acc': 100. * correct / total
            })
        
        return total_loss / len(self.train_loader), correct / total
        
    # Validation
    def validate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc='Validating')
            for inputs, labels in progress_bar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                progress_bar.set_postfix({
                    'loss': total_loss / (progress_bar.n + 1),
                    'acc': 100. * correct / total
                })
        
        return total_loss / len(self.val_loader), correct / total

    # Evaluation
    def evaluate_model(self):
        """Evaluate model performance on validation set"""
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        print("\nEvaluating model on validation set...")
        with torch.no_grad():
            for inputs, labels in tqdm(self.val_loader, desc='Evaluating'):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(all_labels, all_predictions))
        
        # Plot confusion matrix
        self.plot_confusion_matrix(all_labels, all_predictions)
        
        # Plot classification report heatmap
        self.plot_classification_report(all_labels, all_predictions)
        
        return all_labels, all_predictions

    def plot_confusion_matrix(self, all_labels, all_predictions):
        """Plot confusion matrix"""
        cm = confusion_matrix(all_labels, all_predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                   xticklabels=["Real", "Fake"], 
                   yticklabels=["Real", "Fake"])
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title("Confusion Matrix")
        
        # Save plot with timestamp
        cm_path = os.path.join(self.run_dir, f'confusion_matrix_{self.timestamp}.png')
        plt.savefig(cm_path)
        plt.close()
        print(f"\nConfusion matrix saved to {cm_path}")

    def plot_classification_report(self, all_labels, all_predictions):
        """Plot classification report as heatmap"""
        report = classification_report(all_labels, all_predictions, output_dict=True)
        df_report = pd.DataFrame(report).transpose()
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(df_report.iloc[:-1, :-1], annot=True, fmt=".2f", cmap="Blues")
        plt.title("Classification Report")
        
        # Save plot
        report_path = os.path.join(self.run_dir, f'classification_report_{self.timestamp}.png')
        plt.savefig(report_path)
        plt.close()
        print(f"Classification report saved to {report_path}")

    def train(self):
        print(f"Training on {self.device}")
        
        for epoch in range(self.config['epochs']):
            print(f"\nEpoch {epoch+1}/{self.config['epochs']}")
            
            # Train
            train_loss, train_acc = self.train_epoch()
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            
            # Validate
            val_loss, val_acc = self.validate()
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Save best model
            if val_acc > self.history['best_val_acc']:
                self.history['best_val_acc'] = val_acc
                self.save_model('best_model.pth')
            
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
        
        # Save final model and training history
        self.save_model('final_model.pth')
        self.save_history()
        self.plot_history()
        
        # Evaluate final model
        print("\nEvaluating final model...")
        self.evaluate_model()

    def save_model(self, filename):
        """Save the trained model"""
        # Add timestamp to filename
        filename_with_timestamp = f"{os.path.splitext(filename)[0]}_{self.timestamp}.pth"
        save_path = os.path.join(self.run_dir, filename_with_timestamp)
        
        state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'history': self.history
        }
        
        torch.save(state, save_path)
        print(f"Model saved to {save_path}")

    def save_history(self):
        """Save training history"""
        history_path = os.path.join(self.run_dir, f'training_history_{self.timestamp}.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f)

    def plot_history(self):
        """Plot training history"""
        plt.figure(figsize=(12, 4))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Val Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(self.history['train_acc'], label='Train Acc')
        plt.plot(self.history['val_acc'], label='Val Acc')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Save plot
        plt.tight_layout()
        plot_path = os.path.join(self.run_dir, f'training_plot_{self.timestamp}.png')
        plt.savefig(plot_path)
        plt.close()

    def setup_weighted_loss(self, labels):
        """Enhanced weighted loss with focal loss option."""
        import torch.nn.functional as F
        
        class_counts = np.bincount(labels)
        total = len(labels)
        
        # Calculate inverse frequency weights
        class_weights = torch.FloatTensor([
            total / (len(class_counts) * count) for count in class_counts
        ])
        
        # Normalize weights
        class_weights = class_weights / class_weights.sum()
        class_weights = class_weights.to(self.device)
        
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

    def get_augmented_transforms(self, is_minority_class=False):
        """Enhanced transforms with face-specific augmentations."""
        base_transforms = [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation((-10, 10)),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
        ]
        
        if is_minority_class:
            # Additional augmentations for minority class
            advanced_transforms = [
                transforms.RandomAffine(
                    degrees=(-15, 15),
                    translate=(0.1, 0.1),
                    scale=(0.9, 1.1)
                ),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
                transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
            ]
            base_transforms.extend(advanced_transforms)
        
        # Always add the normalization transforms at the end
        base_transforms.extend([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        return transforms.Compose(base_transforms)

    def create_balanced_dataloader(self, dataset, labels):
        """Enhanced balanced dataloader with improved sampling strategy."""
        class_counts = np.bincount(labels)
        max_count = max(class_counts)
        
        # Calculate sample weights for balanced sampling
        class_weights = torch.FloatTensor([max_count / count for count in class_counts])
        sample_weights = class_weights[labels]
        
        # Create sampler with replacement for minority class
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(dataset),
            replacement=True
        )
        
        # Create dataloader with the sampler
        dataloader = DataLoader(
            dataset,
            batch_size=self.config['batch_size'],
            sampler=sampler,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True,
            drop_last=True
        )
        
        return dataloader

    def monitor_class_distribution(self, loader, phase='train'):
        """Monitor class distribution in batches."""
        class_counts = torch.zeros(2)
        for _, labels in loader:
            for label in labels:
                class_counts[label] += 1
        
        total = class_counts.sum()
        distribution = class_counts / total
        
        print(f"\n{phase.capitalize()} Set Class Distribution:")
        print(f"Real: {distribution[0]:.2%}")
        print(f"Fake: {distribution[1]:.2%}")
        
        return distribution

if __name__ == "__main__":
    # Training configuration
    config = {
        'data_dir': 'dataset',          # Path to dataset directory
        'save_dir': 'models',           # Directory to save models and results
        'batch_size': 24,               # Reduced batch size for b2
        'learning_rate': 0.001,
        'epochs': 5,
        'frames_per_video': 32,         # Number of frames to extract from each video
        'clean_start': True,            # Whether to clean processed faces before starting
        'model_version': 'efficientnet-b2'  # Using EfficientNet-B2
    }
    
    # Create trainer and start training
    trainer = DeepfakeTrainer(config)
    trainer.prepare_data()  # This now includes automatic cleanup
    trainer.train()
