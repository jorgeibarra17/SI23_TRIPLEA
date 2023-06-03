import matplotlib.pyplot as plt
import cv2
from network import Network
import torch
from pathlib import Path
import numpy as np
import pandas as pd
from dataset import get_transform, Signal_map
file_path = Path(__file__).parent.absolute()

def to_numpy(tensor: torch.tensor, roll_dims = True):
    '''
    Convert tensor to numpy array
    args:
        - tensor (torch.tensor): tensor to convert
            size: (C, H, W)
    returns:
        - array (np.ndarray): converted array
            size: (H, W, C)
    '''
    if roll_dims:
        if len(tensor.shape) > 3:
            tensor = tensor.squeeze(0) # (1, C, H, W) -> (C, H, W)
        tensor = tensor.permute(1, 2, 0) # (C, H, W) -> (H, W, C)
    array = tensor.detach().cpu().numpy()
    return array

def load_img(path):
    if isinstance(path, str):
        path = Path(path)
    assert path.is_file(), f"El archivo {path} no existe"
    img = cv2.imread(path.as_posix())
    transform = get_transform()
    return transform(img)

def add_img_text(img: np.ndarray, text_label: str):
    '''
    Add text to image
    args:
        - img (np.ndarray): image to add text to
            - size: (C, H, W)
        - text (str): text to add to image
    '''
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    fontColor = (255, 0, 0)
    thickness = 2

    # For the text background
    # Finds space required by the text so that we can put a background with that amount of width.
    (text_w, text_h), _ = cv2.getTextSize(text_label, font, fontScale, thickness)

    # Center text
    x1, y1 = 0, text_h  # Top left corner
    img = cv2.rectangle(img,
                        (x1, y1 - 20),
                        (x1 + text_w, y1),
                        (255, 255, 255),
                        -1)
    if img.shape[-1] == 1 or len(img.shape) == 2: # Grayscale image
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = cv2.putText(img, text_label,
                      (x1, y1),
                      font, fontScale, fontColor, thickness)
    return img

def predict(img_title_paths):
    # TODO: Carga tu modelo 
    model = Network(43)
    model.load_model("modelo_PF.pt")
    for path in img_title_paths:
        #Cargar imagen
        im_file = (file_path / path).as_posix()
        transformed_img = load_img(im_file)
        transformed_img = transformed_img.unsqueeze(0)
        transformed_img = transformed_img.cuda()
        
        #inferencia
        pred_label = model(transformed_img)
        pred_label = torch.argmax(pred_label, dim=-1)
        pred_label = Signal_map[pred_label.item()]
        h, w = transformed_img.shape[:2]
        resize_value = 300
        transformed_img = to_numpy(transformed_img)
        img = cv2.resize(transformed_img, (resize_value//h, resize_value))
        img = add_img_text(img, f"Pred: {pred_label}")
        cv2.imshow("Prediccion", img)
        cv2.waitKey(0)


if __name__ == "__main__":
    img_path = ["./data/crop_dataset/crop_dataset/00000/00000_00000.jpg"]
    predict(img_path)