import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pathlib

file_path = pathlib.Path(__file__).parent.absolute()

class Network(nn.Module):
    def __init__(self, input_dim: int, n_classes: int) -> None:
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # TODO: Calcular dimension de salida
        out_dim = self.calc_out_dim(input_dim, kernel_size=3)
        # TODO: Define las capas de tu red
        self.conv1 = nn.Conv2d(1,48,kernel_size=3)
        self.conv2 = nn.Conv2d(48,64,kernel_size=5)
        self.max_pool1 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(out_dim*out_dim*64,512)
        self.fc2 = nn.Linear(512,n_classes)
        self.to(self.device)
 
    def calc_out_dim(self, in_dim, kernel_size, stride=1, padding=0):
        out_dim = math.floor((in_dim + 2*padding - (kernel_size - 1) - 1/stride))
        out_dim+=1
        return out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Define la propagacion hacia adelante de tu red
        feature_map = self.conv1(x)
        feature_map = F.relu(feature_map)
        feature_map = self.conv2(feature_map)
        feature_map = F.relu(feature_map)
        feature_map = self.max_pool1(feature_map)

        features = torch.flatten(feature_map,start_dim=1)
        features = self.fc1(features)
        features = F.relu(features)
        logits = self.fc2(features)
        proba = F.softmax(logits,dim = -1)
        return logits, proba

    def predict(self, x):
        with torch.inference_mode():
            return self.forward(x)

    def save_model(self, model_name: str):
        '''
            Guarda el modelo en el path especificado
            args:
            - net: definición de la red neuronal (con nn.Sequential o la clase anteriormente definida)
            - path (str): path relativo donde se guardará el modelo
        '''
        models_path = file_path / 'models' / model_name
        # TODO: Guarda los pesos de tu red neuronal en el path especificado
        torch.save(models_path)

    def load_model(self, model_name: str):
        '''
            Carga el modelo en el path especificado
            args:
            - path (str): path relativo donde se guardó el modelo
        '''
        # TODO: Carga los pesos de tu red neuronal
