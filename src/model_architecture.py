import json
from tensorflow.keras import layers, models # type: ignore
from tensorflow.keras.utils import plot_model # type: ignore



def load_model_summary(file_path="model_summary.json"):
    """
    Carrega o resumo do modelo a partir de um arquivo JSON.
    """
    try:
        with open(file_path, "r") as f:
            summary = json.load(f)
        return summary
    except FileNotFoundError:
        return "Resumo do modelo não encontrado."

def create_model(input_shape=(32, 32, 3), summary_file="model_summary.json"):
    """
    Cria uma CNN para classificação binária e salva o resumo do modelo em um arquivo JSON.
    """
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.GlobalAveragePooling2D(),  # Usando Global Average Pooling
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(2, activation='softmax')
    ])
    
    # Captura o resumo do modelo como texto
    summary_str = []
    model.summary(print_fn=lambda x: summary_str.append(x))
    
    # Salva o resumo em um arquivo JSON
    with open(summary_file, "w") as f:
        json.dump("\n".join(summary_str), f)
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
