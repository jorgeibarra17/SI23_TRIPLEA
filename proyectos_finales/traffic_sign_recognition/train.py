from dataset import get_dataloaders
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from network import Network
from plot_losses import PlotLosses
from pathlib import Path

def eval(val_loader, net, cost_function):
    val_loss = []
    for i, batch in enumerate(val_loader, 0):
        batch_imgs, batch_labels = batch
        device = net.device
        batch_imgs = batch_imgs.to(device)
        batch_labels = batch_labels.to(device)
        with torch.inference_mode():
            predictions = net(batch_imgs)
            loss = cost_function(predictions, batch_labels)
            val_loss.append(loss.item())
    return np.mean(val_loss)        

def train():
    # Hyperparámetros
    learning_rate = 1e-5
    num_epochs = 50
    batch_size = 64

    # Obtén los dataloaders de entrenamiento y validación
    train_dataloader, val_dataloader = get_dataloaders(batch_size)
    
    # Crea la instancia de tu modelo y mueve los parámetros a la GPU (si está disponible)
    model = Network(input_dim=32, num_classes=43)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define la función de costo y el optimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Crea el objeto para visualizar las curvas de pérdida
    plotter = PlotLosses()
    running_loss = []
    best_epoch_loss = np.inf
    # Entrenamiento
    for epoch in range(num_epochs):
        # Bucle de entrenamiento
        for i, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch}")):
            batch_images, batch_labels = batch
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)
            
            #Zero grad
            optimizer.zero_grad()

            # Realiza el forward pass y calcula el costo
            predictions = model(batch_images)
            loss = criterion(predictions, batch_labels)

            # Realiza la retropropagación y actualiza los parámetros
            loss.backward()
            optimizer.step()

            running_loss.append(loss.item())
        #Costo promedio de entrenamiento
        train_loss = np.mean(running_loss)
        #Costo promedio de validacion
        val_loss = eval(val_dataloader, model, criterion)

        # Muestra las pérdidas
        tqdm.write(f"Epoch {epoch} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

       # TODO guarda el modelo si el costo de validación es menor al mejor costo de validación
        if val_loss < best_epoch_loss :
            PATH = "modelo_PF.pt"
            torch.save(model.state_dict(), PATH)
        plotter.on_epoch_end(epoch, train_loss, val_loss)
    #plotter.on_train_end()

if __name__ == "__main__":
    train()