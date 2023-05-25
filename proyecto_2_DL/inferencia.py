import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from network import Network
import torch
from utils import to_numpy, get_transforms, add_img_text
from dataset import EMOTIONS_MAP
import pathlib

file_path = pathlib.Path(__file__).parent.absolute()

def load_img(path):
    assert os.path.isfile(path), f"El archivo {path} no existe"
    img = cv2.imread(path)
    val_transforms, unnormalize = get_transforms("test", img_size = 48)
    tensor_img = val_transforms(img)
    denormalized = unnormalize(tensor_img)
    return img, tensor_img, denormalized

def predict(img_title_paths):
    '''
        Hace la inferencia de las imagenes
        args:
        - img_title_paths (dict): diccionario con el titulo de la imagen (key) y el path (value)
    '''
    # Cargar el modelo
    modelo = Network(48, 7)
    modelo.load_model("modelo_1.pt")
    for path in img_title_paths:
        # Cargar la imagen
        # np.ndarray, torch.Tensor
        im_file = (file_path / path).as_posix()
        original, transformed, denormalized = load_img(im_file)
        transformed = transformed.unsqueeze(0)
        transformed = transformed.cuda()
        # Inferencia
        # TODO: Para la imagen de entrada, utiliza tu modelo para predecir la clase mas probable
        pred_label = modelo(transformed)
        pred_label = torch.argmax(pred_label, dim=-1)
        pred_label = EMOTIONS_MAP[pred_label.item()]
        # Original / transformada
        # pred_label (str): nombre de la clase predicha
        h, w = original.shape[:2]
        resize_value = 300
        img = cv2.resize(original, (w * resize_value // h, resize_value))
        img = add_img_text(img, f"Pred: {pred_label}")

        # Mostrar la imagen
        denormalized = to_numpy(denormalized)
        denormalized = cv2.resize(denormalized, (resize_value, resize_value))
        cv2.imshow("Predicción - original", img)
        cv2.imshow("Predicción - transformed", denormalized)
        cv2.waitKey(0)

if __name__=="__main__":
    # Direcciones relativas a este archivo
    img_paths = ["./test_imgs/jorge_happy.jpg"]
    predict(img_paths)
    img_paths = ["./test_imgs/jorge_angry.jpg"]
    predict(img_paths)
    img_paths = ["./test_imgs/adrian_disgust.jpg"]
    predict(img_paths)
    img_paths = ["./test_imgs/miguel_sad.jpg"]
    predict(img_paths)
    img_paths = ["./test_imgs/jorge_fear.jpg"]
    predict(img_paths)
    img_paths = ["./test_imgs/adrian_neutral.jpg"]
    predict(img_paths)
    img_paths = ["./test_imgs/jorge_surprise.jpg"]
    predict(img_paths)