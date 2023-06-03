import torch.nn as nn
import torch
import torch.nn.functional as F
from pathlib import Path

file_path = Path(__file__).parent.absolute()

class Network(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        #Red neuronal
        self.conv1 = nn.Conv2d(3, input_dim, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.to(self.device)

    def forward(self, x: torch.Tensor):
        #Fowardpass
        feature_map = self.conv1(x)
        feature_map = F.relu(feature_map)
        feature_map = self.conv2(feature_map)
        feature_map = self.pool(feature_map)
        
        features = torch.flatten(feature_map, start_dim=-1)
        features = self.fc1(features)
        features = F.relu(features)
        features = self.fc2(features)
        
        return features
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            return self.forward(x)
        
    def save_model(self, model_name: str):
        models_path = file_path/'models'/model_name
        torch.save(models_path)
        
    def load_model(self, model_name: str):
        torch.load(model_name)        