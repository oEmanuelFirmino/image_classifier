import numpy as np
import tensorflow as tf
from PIL import Image

class DogCatPredictor:
    def __init__(self, model_path):
        """
        Inicializa o predictor carregando o modelo salvo
        """
        self.model = tf.keras.models.load_model(model_path)
        self.target_size = (32, 32)
        self.classes = ['Gato', 'Cachorro']

    def preprocess_image(self, image_path):
        """
        Pré-processa uma única imagem para predição
        """
        with Image.open(image_path) as img:
            img = img.convert('RGB')
            img = img.resize(self.target_size)
            img_array = np.array(img)
            img_array = img_array.astype('float32') / 255.0
            return np.expand_dims(img_array, axis=0)

    def predict(self, image_path):
        """
        Realiza a predição em uma única imagem
        """
        processed_image = self.preprocess_image(image_path)
        prediction = self.model.predict(processed_image)
        class_idx = np.argmax(prediction[0])
        confidence = prediction[0][class_idx]
        
        return {
            'classe': self.classes[class_idx],
            'confianca': float(confidence),
            'probabilidades': {
                classe: float(prob)
                for classe, prob in zip(self.classes, prediction[0])
            }
        }

    def predict_batch(self, image_paths):
        """
        Realiza predições em um lote de imagens
        """
        return [self.predict(img_path) for img_path in image_paths]