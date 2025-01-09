import os
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def process_image(image_path, target_size=(32, 32)):
    """
    Processa uma única imagem e retorna array numpy
    """
    try:
        with Image.open(image_path) as img:
            img = img.convert('RGB')
            img = img.resize(target_size)
            return np.array(img)
    except (IOError, OSError) as e:
        print(f"Erro ao processar {image_path}: {e}")
        return None

def load_dataset(dog_path, cat_path, target_size=(32, 32)):
    """
    Carrega e processa o dataset completo usando processamento paralelo
    """
    dog_images = [os.path.join(dog_path, f) for f in os.listdir(dog_path)]
    cat_images = [os.path.join(cat_path, f) for f in os.listdir(cat_path)]
    
    images = dog_images + cat_images
    classes = [[0, 1]] * len(dog_images) + [[1, 0]] * len(cat_images)
    
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(
            executor.map(process_image, images),
            total=len(images),
            desc="Processando imagens"
        ))
    
    valid_results = [(img, cls) for img, cls in zip(results, classes) if img is not None]
    
    if not valid_results:
        raise ValueError("Nenhuma imagem válida encontrada")
    
    train_images, train_classes = zip(*valid_results)
    return np.array(train_images), np.array(train_classes)

def prepare_data(images, classes, test_size=0.2):
    """
    Prepara os dados para treinamento, normalizando e dividindo em treino/validação
    """
    images = images.astype('float32') / 255.0
    return train_test_split(images, classes, test_size=test_size, random_state=42)