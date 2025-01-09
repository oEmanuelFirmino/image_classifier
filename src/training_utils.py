import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint  # type: ignore
import json
import os


def load_training_history(file_path="training_history.json"):
    """
    Carrega o histórico de treinamento de um arquivo JSON.
    """
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            history = json.load(f)
        return history
    else:
        raise FileNotFoundError(f"O arquivo {file_path} não foi encontrado.")


def get_callbacks(model_path):
    """
    Configura callbacks para treinamento
    """
    return [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ModelCheckpoint(
            model_path, monitor="val_accuracy", save_best_only=True, mode="max"
        ),
    ]


def plot_training_history(history):
    """
    Plota o histórico de treinamento
    """
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Treino")
    plt.plot(history.history["val_accuracy"], label="Validação")
    plt.title("Acurácia do Modelo")
    plt.xlabel("Época")
    plt.ylabel("Acurácia")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Treino")
    plt.plot(history.history["val_loss"], label="Validação")
    plt.title("Loss do Modelo")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.show()


def save_training_history(history, file_path="training_history.json"):
    """
    Salva o histórico de treinamento em um arquivo JSON.
    """
    history_dict = {
        "accuracy": history.history["accuracy"],
        "val_accuracy": history.history["val_accuracy"],
        "loss": history.history["loss"],
        "val_loss": history.history["val_loss"],
    }

    with open(file_path, "w") as f:
        json.dump(history_dict, f)