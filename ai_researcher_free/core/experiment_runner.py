"""
Free Experiment Runner
Runs ML experiments using free compute resources and datasets
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, random_split
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import time
import logging
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import pickle

logger = logging.getLogger(__name__)

class FreeExperimentRunner:
    """
    Run machine learning experiments using only free resources
    - CPU-only PyTorch (free)
    - Small datasets (CIFAR-10, MNIST)
    - Lightweight models
    - Google Colab integration ready
    """
    
    def __init__(self, use_gpu: bool = None):
        self.device = self._setup_device(use_gpu)
        self.results_dir = "results"
        self.models_dir = "models"
        self.data_dir = "data"
        
        # Create directories
        for dir_path in [self.results_dir, self.models_dir, self.data_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        logger.info(f"Experiment runner initialized with device: {self.device}")
    
    def _setup_device(self, use_gpu: Optional[bool]) -> torch.device:
        """Setup compute device (CPU/GPU)"""
        if use_gpu is None:
            # Auto-detect
            if torch.cuda.is_available():
                device = torch.device('cuda')
                logger.info("GPU detected and will be used")
            else:
                device = torch.device('cpu')
                logger.info("Using CPU (GPU not available)")
        elif use_gpu and torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        
        return device
    
    def run_vision_experiment(self, hypothesis: str, 
                            experiment_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Run a computer vision experiment based on hypothesis
        
        Args:
            hypothesis: Research hypothesis to test
            experiment_config: Optional experiment configuration
            
        Returns:
            Experiment results dictionary
        """
        logger.info(f"Starting vision experiment: {hypothesis}")
        
        # Default configuration
        config = {
            'dataset': 'cifar10',
            'model_type': 'cnn',
            'batch_size': 32,
            'epochs': 10,
            'learning_rate': 0.001,
            'train_subset_size': 1000,  # Small subset for free compute
            'test_subset_size': 200
        }
        
        if experiment_config:
            config.update(experiment_config)
        
        # Load dataset
        train_loader, test_loader, num_classes = self._load_vision_dataset(config)
        
        # Create model based on hypothesis
        model = self._create_vision_model(hypothesis, num_classes, config['model_type'])
        model = model.to(self.device)
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
        
        # Train model
        training_history = self._train_model(
            model, train_loader, criterion, optimizer, config['epochs']
        )
        
        # Evaluate model
        test_results = self._evaluate_model(model, test_loader, criterion)
        
        # Generate results
        results = {
            'hypothesis': hypothesis,
            'config': config,
            'training_history': training_history,
            'test_results': test_results,
            'model_info': self._get_model_info(model),
            'timestamp': datetime.now().isoformat(),
            'device_used': str(self.device)
        }
        
        # Save results
        self._save_experiment_results(results, f"vision_experiment_{int(time.time())}")
        
        logger.info(f"Vision experiment completed. Test accuracy: {test_results['accuracy']:.2f}%")
        return results
    
    def _load_vision_dataset(self, config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, int]:
        """Load vision dataset (free datasets only)"""
        dataset_name = config['dataset'].lower()
        
        # Define transforms
        if dataset_name == 'cifar10':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            
            # Download CIFAR-10 (free)
            trainset = torchvision.datasets.CIFAR10(
                root=self.data_dir, train=True, download=True, transform=transform
            )
            testset = torchvision.datasets.CIFAR10(
                root=self.data_dir, train=False, download=True, transform=transform
            )
            num_classes = 10
            
        elif dataset_name == 'mnist':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            
            trainset = torchvision.datasets.MNIST(
                root=self.data_dir, train=True, download=True, transform=transform
            )
            testset = torchvision.datasets.MNIST(
                root=self.data_dir, train=False, download=True, transform=transform
            )
            num_classes = 10
            
        else:
            raise ValueError(f"Dataset {dataset_name} not supported. Use 'cifar10' or 'mnist'")
        
        # Create subsets for faster experimentation
        train_subset = Subset(trainset, 
                             np.random.choice(len(trainset), 
                                            min(config['train_subset_size'], len(trainset)), 
                                            replace=False))
        test_subset = Subset(testset,
                           np.random.choice(len(testset),
                                          min(config['test_subset_size'], len(testset)),
                                          replace=False))
        
        # Create data loaders
        train_loader = DataLoader(train_subset, batch_size=config['batch_size'], shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=config['batch_size'], shuffle=False)
        
        logger.info(f"Loaded {dataset_name}: {len(train_subset)} train, {len(test_subset)} test samples")
        return train_loader, test_loader, num_classes
    
    def _create_vision_model(self, hypothesis: str, num_classes: int, model_type: str) -> nn.Module:
        """Create vision model based on hypothesis"""
        hypothesis_lower = hypothesis.lower()
        
        if 'attention' in hypothesis_lower:
            return self._create_attention_model(num_classes)
        elif 'transformer' in hypothesis_lower or 'vit' in hypothesis_lower:
            return self._create_simple_transformer(num_classes)
        elif 'resnet' in hypothesis_lower:
            return self._create_simple_resnet(num_classes)
        else:
            return self._create_simple_cnn(num_classes)
    
    def _create_simple_cnn(self, num_classes: int) -> nn.Module:
        """Create simple CNN model"""
        class SimpleCNN(nn.Module):
            def __init__(self, num_classes):
                super(SimpleCNN, self).__init__()
                self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
                self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
                self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
                self.pool = nn.MaxPool2d(2, 2)
                self.fc1 = nn.Linear(128 * 4 * 4, 512)
                self.fc2 = nn.Linear(512, num_classes)
                self.dropout = nn.Dropout(0.5)
                
            def forward(self, x):
                x = self.pool(torch.relu(self.conv1(x)))
                x = self.pool(torch.relu(self.conv2(x)))
                x = self.pool(torch.relu(self.conv3(x)))
                x = x.view(-1, 128 * 4 * 4)
                x = torch.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.fc2(x)
                return x
        
        return SimpleCNN(num_classes)
    
    def _create_attention_model(self, num_classes: int) -> nn.Module:
        """Create CNN with attention mechanism"""
        class AttentionCNN(nn.Module):
            def __init__(self, num_classes):
                super(AttentionCNN, self).__init__()
                self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
                self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
                self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
                self.pool = nn.MaxPool2d(2, 2)
                
                # Simple attention mechanism
                self.attention = nn.MultiheadAttention(128, 4, batch_first=True)
                
                self.fc1 = nn.Linear(128, 512)
                self.fc2 = nn.Linear(512, num_classes)
                self.dropout = nn.Dropout(0.5)
                
            def forward(self, x):
                x = self.pool(torch.relu(self.conv1(x)))
                x = self.pool(torch.relu(self.conv2(x)))
                x = self.pool(torch.relu(self.conv3(x)))
                
                # Reshape for attention
                b, c, h, w = x.shape
                x = x.view(b, c, h*w).transpose(1, 2)  # (batch, seq_len, features)
                
                # Apply attention
                x, _ = self.attention(x, x, x)
                x = x.mean(dim=1)  # Global average pooling
                
                x = torch.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.fc2(x)
                return x
        
        return AttentionCNN(num_classes)
    
    def _create_simple_transformer(self, num_classes: int) -> nn.Module:
        """Create simple vision transformer"""
        class SimpleViT(nn.Module):
            def __init__(self, num_classes, patch_size=4, embed_dim=128):
                super(SimpleViT, self).__init__()
                self.patch_size = patch_size
                self.embed_dim = embed_dim
                
                # Patch embedding
                self.patch_embed = nn.Conv2d(3, embed_dim, patch_size, patch_size)
                
                # Transformer layers
                self.transformer = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=embed_dim,
                        nhead=4,
                        dim_feedforward=256,
                        batch_first=True
                    ),
                    num_layers=3
                )
                
                # Classification head
                self.classifier = nn.Linear(embed_dim, num_classes)
                
            def forward(self, x):
                # Create patches
                x = self.patch_embed(x)  # (batch, embed_dim, h/patch_size, w/patch_size)
                b, c, h, w = x.shape
                x = x.reshape(b, c, h*w).transpose(1, 2)  # (batch, seq_len, embed_dim)
                
                # Add positional encoding (simplified)
                seq_len = x.shape[1]
                pos_encoding = torch.arange(seq_len, device=x.device).unsqueeze(0).unsqueeze(-1)
                pos_encoding = pos_encoding.expand(b, seq_len, c).float() * 0.01
                x = x + pos_encoding
                
                # Apply transformer
                x = self.transformer(x)
                
                # Global average pooling and classification
                x = x.mean(dim=1)
                return self.classifier(x)
        
        return SimpleViT(num_classes)
    
    def _create_simple_resnet(self, num_classes: int) -> nn.Module:
        """Create simple ResNet-inspired model"""
        class SimpleResNet(nn.Module):
            def __init__(self, num_classes):
                super(SimpleResNet, self).__init__()
                self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
                self.bn1 = nn.BatchNorm2d(32)
                
                # Residual blocks
                self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
                self.bn2 = nn.BatchNorm2d(32)
                self.conv3 = nn.Conv2d(32, 64, 3, padding=1, stride=2)
                self.bn3 = nn.BatchNorm2d(64)
                self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
                self.bn4 = nn.BatchNorm2d(64)
                
                self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
                self.fc = nn.Linear(64, num_classes)
                
            def forward(self, x):
                # First conv
                x = torch.relu(self.bn1(self.conv1(x)))
                
                # Residual block 1
                identity = x
                x = torch.relu(self.bn2(self.conv2(x)))
                x = x + identity  # Skip connection
                
                # Downsample and residual block 2
                x = torch.relu(self.bn3(self.conv3(x)))
                identity = x
                x = torch.relu(self.bn4(self.conv4(x)))
                x = x + identity  # Skip connection
                
                # Global average pooling and classification
                x = self.avgpool(x)
                x = x.view(x.size(0), -1)
                return self.fc(x)
        
        return SimpleResNet(num_classes)
    
    def _train_model(self, model: nn.Module, train_loader: DataLoader,
                    criterion, optimizer, epochs: int) -> List[Dict[str, float]]:
        """Train the model and return training history"""
        model.train()
        history = []
        
        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            
            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            epoch_loss = running_loss / len(train_loader)
            epoch_acc = 100. * correct / total
            
            history.append({
                'epoch': epoch + 1,
                'loss': epoch_loss,
                'accuracy': epoch_acc
            })
            
            if (epoch + 1) % 2 == 0:
                logger.info(f'Epoch [{epoch+1}/{epochs}] Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}%')
        
        return history
    
    def _evaluate_model(self, model: nn.Module, test_loader: DataLoader, 
                       criterion) -> Dict[str, float]:
        """Evaluate model on test set"""
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return {
            'loss': test_loss / len(test_loader),
            'accuracy': 100. * correct / total,
            'total_samples': total
        }
    
    def _get_model_info(self, model: nn.Module) -> Dict[str, Any]:
        """Get model information"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'model_class': model.__class__.__name__,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
        }
    
    def _save_experiment_results(self, results: Dict[str, Any], experiment_name: str):
        """Save experiment results"""
        # Save JSON results
        json_path = os.path.join(self.results_dir, f"{experiment_name}.json")
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save model if requested
        # model_path = os.path.join(self.models_dir, f"{experiment_name}.pth")
        # torch.save(model.state_dict(), model_path)
        
        logger.info(f"Results saved to {json_path}")
    
    def run_nlp_experiment(self, hypothesis: str, 
                          experiment_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run NLP experiment (simplified for free resources)"""
        logger.info(f"Starting NLP experiment: {hypothesis}")
        
        # For free resources, we'll use simple text classification
        config = {
            'task': 'sentiment_classification',
            'vocab_size': 10000,
            'embed_dim': 128,
            'hidden_dim': 256,
            'num_classes': 2,
            'max_length': 100,
            'batch_size': 32,
            'epochs': 5,
            'learning_rate': 0.001
        }
        
        if experiment_config:
            config.update(experiment_config)
        
        # Create simple synthetic dataset for demonstration
        train_data, test_data = self._create_synthetic_text_data(config)
        
        # Create model
        model = self._create_nlp_model(hypothesis, config)
        model = model.to(self.device)
        
        # Train and evaluate
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
        
        training_history = self._train_nlp_model(model, train_data, criterion, optimizer, config['epochs'])
        test_results = self._evaluate_nlp_model(model, test_data, criterion)
        
        results = {
            'hypothesis': hypothesis,
            'config': config,
            'training_history': training_history,
            'test_results': test_results,
            'model_info': self._get_model_info(model),
            'timestamp': datetime.now().isoformat(),
            'device_used': str(self.device)
        }
        
        self._save_experiment_results(results, f"nlp_experiment_{int(time.time())}")
        return results
    
    def _create_synthetic_text_data(self, config: Dict[str, Any]) -> Tuple[List, List]:
        """Create synthetic text data for experimentation"""
        vocab_size = config['vocab_size']
        max_length = config['max_length']
        
        # Generate random sequences (simulating tokenized text)
        train_size = 1000
        test_size = 200
        
        train_data = []
        for _ in range(train_size):
            seq_len = np.random.randint(10, max_length)
            sequence = torch.randint(1, vocab_size, (seq_len,))
            label = torch.randint(0, config['num_classes'], (1,)).item()
            train_data.append((sequence, label))
        
        test_data = []
        for _ in range(test_size):
            seq_len = np.random.randint(10, max_length)
            sequence = torch.randint(1, vocab_size, (seq_len,))
            label = torch.randint(0, config['num_classes'], (1,)).item()
            test_data.append((sequence, label))
        
        return train_data, test_data
    
    def _create_nlp_model(self, hypothesis: str, config: Dict[str, Any]) -> nn.Module:
        """Create NLP model based on hypothesis"""
        hypothesis_lower = hypothesis.lower()
        
        if 'attention' in hypothesis_lower or 'transformer' in hypothesis_lower:
            return self._create_simple_transformer_nlp(config)
        elif 'lstm' in hypothesis_lower or 'rnn' in hypothesis_lower:
            return self._create_lstm_model(config)
        else:
            return self._create_simple_nlp_model(config)
    
    def _create_simple_nlp_model(self, config: Dict[str, Any]) -> nn.Module:
        """Create simple NLP model"""
        class SimpleNLPModel(nn.Module):
            def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
                super(SimpleNLPModel, self).__init__()
                self.embedding = nn.Embedding(vocab_size, embed_dim)
                self.fc1 = nn.Linear(embed_dim, hidden_dim)
                self.fc2 = nn.Linear(hidden_dim, num_classes)
                self.dropout = nn.Dropout(0.5)
                
            def forward(self, x):
                x = self.embedding(x)  # (batch, seq_len, embed_dim)
                x = x.mean(dim=1)  # Simple average pooling
                x = torch.relu(self.fc1(x))
                x = self.dropout(x)
                return self.fc2(x)
        
        return SimpleNLPModel(
            config['vocab_size'], config['embed_dim'], 
            config['hidden_dim'], config['num_classes']
        )
    
    def _create_lstm_model(self, config: Dict[str, Any]) -> nn.Module:
        """Create LSTM-based model"""
        class LSTMModel(nn.Module):
            def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
                super(LSTMModel, self).__init__()
                self.embedding = nn.Embedding(vocab_size, embed_dim)
                self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
                self.fc = nn.Linear(hidden_dim, num_classes)
                self.dropout = nn.Dropout(0.5)
                
            def forward(self, x):
                x = self.embedding(x)
                lstm_out, (hidden, _) = self.lstm(x)
                # Use last hidden state
                x = hidden[-1]  # Take last layer's hidden state
                x = self.dropout(x)
                return self.fc(x)
        
        return LSTMModel(
            config['vocab_size'], config['embed_dim'],
            config['hidden_dim'], config['num_classes']
        )
    
    def _create_simple_transformer_nlp(self, config: Dict[str, Any]) -> nn.Module:
        """Create simple transformer for NLP"""
        class SimpleTransformerNLP(nn.Module):
            def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
                super(SimpleTransformerNLP, self).__init__()
                self.embedding = nn.Embedding(vocab_size, embed_dim)
                self.transformer = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=embed_dim,
                        nhead=4,
                        dim_feedforward=hidden_dim,
                        batch_first=True
                    ),
                    num_layers=2
                )
                self.fc = nn.Linear(embed_dim, num_classes)
                
            def forward(self, x):
                x = self.embedding(x)
                x = self.transformer(x)
                x = x.mean(dim=1)  # Global average pooling
                return self.fc(x)
        
        return SimpleTransformerNLP(
            config['vocab_size'], config['embed_dim'],
            config['hidden_dim'], config['num_classes']
        )
    
    def _train_nlp_model(self, model: nn.Module, train_data: List,
                        criterion, optimizer, epochs: int) -> List[Dict[str, float]]:
        """Train NLP model"""
        model.train()
        history = []
        
        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            
            # Simple batch processing
            np.random.shuffle(train_data)
            
            for i in range(0, len(train_data), 32):
                batch = train_data[i:i+32]
                
                # Pad sequences to same length
                sequences = [item[0] for item in batch]
                labels = torch.tensor([item[1] for item in batch])
                
                # Simple padding
                max_len = max(len(seq) for seq in sequences)
                padded_sequences = torch.zeros(len(sequences), max_len, dtype=torch.long)
                for j, seq in enumerate(sequences):
                    padded_sequences[j, :len(seq)] = seq
                
                inputs = padded_sequences.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            epoch_loss = running_loss / (len(train_data) // 32 + 1)
            epoch_acc = 100. * correct / total
            
            history.append({
                'epoch': epoch + 1,
                'loss': epoch_loss,
                'accuracy': epoch_acc
            })
            
            logger.info(f'NLP Epoch [{epoch+1}/{epochs}] Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}%')
        
        return history
    
    def _evaluate_nlp_model(self, model: nn.Module, test_data: List, criterion) -> Dict[str, float]:
        """Evaluate NLP model"""
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for i in range(0, len(test_data), 32):
                batch = test_data[i:i+32]
                
                sequences = [item[0] for item in batch]
                labels = torch.tensor([item[1] for item in batch])
                
                max_len = max(len(seq) for seq in sequences)
                padded_sequences = torch.zeros(len(sequences), max_len, dtype=torch.long)
                for j, seq in enumerate(sequences):
                    padded_sequences[j, :len(seq)] = seq
                
                inputs = padded_sequences.to(self.device)
                labels = labels.to(self.device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return {
            'loss': test_loss / (len(test_data) // 32 + 1),
            'accuracy': 100. * correct / total,
            'total_samples': total
        }
    
    def create_experiment_report(self, results: Dict[str, Any]) -> str:
        """Create a formatted experiment report"""
        report = f"""
# Experiment Report

## Hypothesis
{results['hypothesis']}

## Configuration
- Dataset: {results['config'].get('dataset', 'synthetic')}
- Model: {results['model_info']['model_class']}
- Device: {results['device_used']}
- Parameters: {results['model_info']['total_parameters']:,}

## Results
- Final Test Accuracy: {results['test_results']['accuracy']:.2f}%
- Final Test Loss: {results['test_results']['loss']:.4f}
- Model Size: {results['model_info']['model_size_mb']:.2f} MB

## Training Progress
"""
        
        for epoch_data in results['training_history'][-5:]:  # Last 5 epochs
            report += f"- Epoch {epoch_data['epoch']}: {epoch_data['accuracy']:.2f}% accuracy, {epoch_data['loss']:.4f} loss\n"
        
        report += f"\n## Timestamp\n{results['timestamp']}\n"
        
        return report
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get summary of all experiments"""
        results_files = [f for f in os.listdir(self.results_dir) if f.endswith('.json')]
        
        experiments = []
        for file in results_files:
            try:
                with open(os.path.join(self.results_dir, file), 'r') as f:
                    result = json.load(f)
                    experiments.append({
                        'file': file,
                        'hypothesis': result.get('hypothesis', 'Unknown'),
                        'accuracy': result.get('test_results', {}).get('accuracy', 0),
                        'timestamp': result.get('timestamp', 'Unknown')
                    })
            except Exception as e:
                logger.error(f"Error reading {file}: {e}")
        
        return {
            'total_experiments': len(experiments),
            'experiments': sorted(experiments, key=lambda x: x['accuracy'], reverse=True),
            'best_accuracy': max([e['accuracy'] for e in experiments]) if experiments else 0
        }
