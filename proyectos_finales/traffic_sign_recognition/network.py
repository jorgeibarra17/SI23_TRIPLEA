import torch.nn as nn
import torch
import torch.nn.functional as F
from pathlib import Path

file_path = Path(__file__).parent.absolute()

class Network(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        #Red neuronal
        self.conv1 = nn.Conv2d(3, 32, kernel_size=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=2)
        self.drop1 = nn.Dropout2d(0.4)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(14*14*128, 32)
        self.drop2 = nn.Dropout2d(0.2)
        self.fc2 = nn.Linear(32, 8)
        self.fc3 = nn.Linear(8, num_classes)
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #Fowardpass
        feature_map = self.conv1(x)
        feature_map = F.relu(feature_map)
        feature_map = self.conv2(feature_map)
        feature_map = F.relu(feature_map)
        feature_map = self.drop1(feature_map)
        feature_map = self.conv3(feature_map)
        feature_map = self.pool2(feature_map)
        
        features = torch.flatten(feature_map, start_dim=1)
        features = self.fc1(features)
        features = F.relu(features)
        features = self.drop2(features)
        features = self.fc2(features)
        features = F.relu(features)
        features = self.fc3(features)
        return features
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            return self.forward(x)
        
    def save_model(self, model_name: str):
        models_path = file_path/'models'/model_name
        torch.save(models_path)
        
    def load_model(self, model_name: str):
        torch.load(model_name)        