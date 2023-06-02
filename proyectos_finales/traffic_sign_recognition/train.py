from dataset import get_dataloaders
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from network import Network
from plot_losses import PlotLosses
from pathlib import Path

def train():
    # Hyperparámetros
    learning_rate = 0.001
    num_epochs = 50
    batch_size = 64

    # Obtén los dataloaders de entrenamiento y validación
    train_loader, val_loader = get_dataloaders(batch_size)

    # Crea la instancia de tu modelo y mueve los parámetros a la GPU (si está disponible)
    model = Network(num_classes=43)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define la función de costo y el optimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Crea el objeto para visualizar las curvas de pérdida
    plotter = PlotLosses()

    # Entrenamiento
    for epoch in range(num_epochs):
        train_loss = 0.0
        val_loss = 0.0

        # Bucle de entrenamiento
        model.train()
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            # Realiza el forward pass y calcula el costo
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Realiza la retropropagación y actualiza los parámetros
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        # Bucle de validación
        model.eval()
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)

        # Calcula las pérdidas promedio por epoch
        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)

        # Muestra las pérdidas
        tqdm.write(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

        # Actualiza el plotter de las curvas de pérdida
        plotter.update(train_loss, val_loss)
        plotter.plot()

    # Guarda el modelo después de finalizar el entrenamiento
    save_path = Path("modelo.pth")
    torch.save(model.state_dict(), save_path)
    tqdm.write(f"Modelo guardado en {save_path}")


if __name__ == "__main__":
    train()