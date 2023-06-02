import torch.nn as nn
import torch
import torch.nn.functional as F
from pathlib import Path

file_path = Path(__file__).parent.absolute()

class Network(nn.Module):
    def __init__(self, num_classes):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), 64 * 8 * 8)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

def main():
    # Definir el número de clases en la base de datos
    num_classes = 43

    # Crear una instancia del modelo
    net = Network(num_classes)

    # Imprimir la arquitectura de la red neuronal
    print(net)

    # Ejecutar una pasada hacia adelante con datos aleatorios
    output = net(torch.rand(1, 3, 32, 32))
    print(output)

if __name__ == "__main__":
    main()