from src.data_processor import load_dataset, prepare_data
from src.model_architecture import create_model
from src.training_utils import get_callbacks, plot_training_history, save_training_history
from src.predictor import DogCatPredictor

def train_model():
    dog_path = "kagglecatsanddogs_5340/PetImages/Dog/"
    cat_path = "kagglecatsanddogs_5340/PetImages/Cat/"
    model_path = 'modelo_cats_dogs.keras'
    
    print("Carregando dataset...")
    images, classes = load_dataset(dog_path, cat_path)
    X_train, X_val, y_train, y_val = prepare_data(images, classes)
    
    print("Criando modelo...")
    model = create_model()
    print(model.summary())
    
    print("Iniciando treinamento...")
    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=get_callbacks(model_path)
    )
    
    plot_training_history(history)

    save_training_history(history, 'training_history.json')

    
    test_loss, test_acc = model.evaluate(X_val, y_val)
    print(f'\nAcurácia no conjunto de validação: {test_acc:.4f}')

if __name__ == "__main__":
    train_model()
