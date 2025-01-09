import streamlit as st
import PIL.Image
import plotly.express as px
import matplotlib.pyplot as plt
from src.model_architecture import create_model, load_model_summary
from src.predictor import DogCatPredictor
from src.training_utils import load_training_history
import os
from tensorflow.keras import layers, models  # type: ignore


def display_model_summary(st):
    """
    Função para exibir o summary do modelo de forma organizada
    """
    st.subheader("🧬 Arquitetura do Modelo")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Camadas Totais", value="14")
    with col2:
        st.metric(label="Parâmetros Treináveis", value="111,426")
    with col3:
        st.metric(label="Tamanho de Entrada", value="32x32x3")

    st.markdown("### 📊 Detalhamento das Camadas")

    data = [
        [
            "1",
            "Conv2D",
            "(None, 30, 30, 32)",
            896,
            "Extração de características básicas",
        ],
        [
            "2",
            "BatchNormalization",
            "(None, 30, 30, 32)",
            128,
            "Normalização dos dados",
        ],
        ["3", "MaxPooling2D", "(None, 15, 15, 32)", 0, "Redução de dimensionalidade"],
        [
            "4",
            "Conv2D",
            "(None, 13, 13, 64)",
            18496,
            "Extração de características intermediárias",
        ],
        [
            "5",
            "BatchNormalization",
            "(None, 13, 13, 64)",
            256,
            "Normalização dos dados",
        ],
        ["6", "MaxPooling2D", "(None, 6, 6, 64)", 0, "Redução de dimensionalidade"],
        [
            "7",
            "Conv2D",
            "(None, 4, 4, 128)",
            73856,
            "Extração de características avançadas",
        ],
        ["8", "BatchNormalization", "(None, 4, 4, 128)", 512, "Normalização dos dados"],
        ["9", "MaxPooling2D", "(None, 2, 2, 128)", 0, "Redução de dimensionalidade"],
        [
            "10",
            "GlobalAveragePooling2D",
            "(None, 128)",
            0,
            "Pooling global dos features",
        ],
        ["11", "Dense", "(None, 128)", 16512, "Camada totalmente conectada"],
        ["12", "BatchNormalization", "(None, 128)", 512, "Normalização dos dados"],
        ["13", "Dropout", "(None, 128)", 0, "Prevenção de overfitting"],
        ["14", "Dense", "(None, 2)", 258, "Camada de classificação final"],
    ]

    import pandas as pd

    df = pd.DataFrame(
        data, columns=["Nº", "Camada", "Shape de Saída", "Parâmetros", "Descrição"]
    )

    st.dataframe(
        df.style.set_properties(
            **{"background-color": "darkblue", "color": "white"}, subset=["Nº"]
        )
        .set_properties(**{}, subset=["Descrição"])
        .format({"Parâmetros": lambda x: f"{x:,}"}),
        height=500,
    )

    st.markdown(
        """
    ### 🔄 Fluxo de Processamento
    
    1. **Entrada**: Imagem RGB 32x32
    2. **Bloco Convolucional 1**: Conv2D → BatchNorm → MaxPool
    3. **Bloco Convolucional 2**: Conv2D → BatchNorm → MaxPool
    4. **Bloco Convolucional 3**: Conv2D → BatchNorm → MaxPool
    5. **Feature Extraction**: GlobalAveragePooling2D
    6. **Classificação**: Dense → BatchNorm → Dropout → Dense
    
    > 💡 **Nota**: A arquitetura usa BatchNormalization após cada camada convolucional para melhorar a estabilidade do treinamento e Dropout para prevenir overfitting.
    """
    )


def main():
    st.set_page_config(
        page_title="Classificador de Cães e Gatos", page_icon="🐾", layout="centered"
    )

    st.title("Classificador de Cães e Gatos 🐶🐱")

    try:
        predictor = DogCatPredictor("modelo_cats_dogs.keras")
    except Exception as e:
        st.error(
            "Erro ao carregar o modelo. Certifique-se de que o arquivo 'modelo_cats_dogs.keras' existe."
        )
        st.stop()

    uploaded_file = st.file_uploader(
        "📤 Faça upload de uma imagem de cachorro ou gato", type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:
        image = PIL.Image.open(uploaded_file)
        st.image(image, caption="📷 Imagem enviada", use_container_width=True)

        if st.button("🔮 Fazer Predição"):
            temp_dir = "temp"
            os.makedirs(temp_dir, exist_ok=True)

            temp_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            try:
                with st.spinner("🐾 Analisando a imagem..."):
                    resultado = predictor.predict(temp_path)

                os.remove(temp_path)

                st.success(f"🎉 Predição concluída!")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(label="📌 Classificação", value=resultado["classe"])

                with col2:
                    st.metric(
                        label="🔍 Confiança", value=f"{resultado['confianca']:.1%}"
                    )

                st.subheader("📊 Probabilidades detalhadas:")
                probabilidade = resultado["probabilidades"]

                fig = px.bar(
                    x=list(probabilidade.keys()),
                    y=list(probabilidade.values()),
                    labels={"x": "Classe", "y": "Probabilidade"},
                    title="Probabilidades de Classificação",
                )
                st.plotly_chart(fig)

            except Exception as e:
                st.error(f"❌ Erro ao processar a imagem: {str(e)}")

    tabs = st.tabs(["ℹ️ Sobre o Projeto", "📈 Sobre o Treinamento"])

    with tabs[0]:
        st.markdown(
            """
        Este é um classificador de imagens que utiliza Deep Learning para identificar se uma imagem contém um **cachorro** ou um **gato**.
        
        **Como usar:**
        1. 📤 Faça upload de uma imagem usando o botão acima.
        2. 🔮 Clique em "Fazer Predição".
        3. ✅ Aguarde o resultado da classificação.
        
        **Tecnologias utilizadas:**
        - 🧠 TensorFlow/Keras para o modelo de Deep Learning.
        - 🌐 Streamlit para a interface web.
        """
        )

    with tabs[1]:  # Aba de gráficos de treinamento
        st.subheader("📈 Acurácia do Modelo:")
        fig_acc, fig_loss = plot_training_history()
        st.pyplot(fig_acc)
        st.pyplot(fig_loss)

        st.subheader("🧠 Arquitetura do Modelo:")
        st.markdown(
            """
            O modelo é uma rede neural convolucional (CNN) projetada para classificar imagens de cães e gatos.
            
            **Camadas do Modelo:**
            1. **Conv2D (32 filtros, 3x3):** Extração de características iniciais, como bordas e texturas.
            2. **MaxPooling2D (2x2):** Redução da dimensionalidade, mantendo as informações mais relevantes.
            3. **Conv2D (64 filtros, 3x3):** Extração de características mais complexas.
            4. **MaxPooling2D (2x2):** Nova redução da dimensionalidade.
            5. **Conv2D (64 filtros, 3x3):** Camada adicional de extração de características.
            6. **Flatten:** Achata a saída para prepará-la para as camadas densas.
            7. **Dense (64 neurônios):** Camada densa com ativação ReLU.
            8. **Dropout (50%):** Regularização para evitar overfitting.
            9. **Dense (2 neurônios):** Camada final para classificação binária (Cachorro ou Gato).
            
            **Função de Ativação:**
            - **ReLU:** Utilizada nas camadas convolucionais e densas para adicionar não-linearidade.
            - **Softmax:** Usada na camada final para obter as probabilidades de cada classe.
            
            **Função de Custo:**
            - **Categorical Crossentropy:** Adequada para problemas de classificação com múltiplas classes.
            
            **Otimização:**
            - **Adam:** Algoritmo de otimização eficiente para treinamento de redes neurais.
            """
        )

        display_model_summary(st)


def plot_training_history():
    data = load_training_history("training_history.json")

    history = {
        "accuracy": data["accuracy"],
        "val_accuracy": data["val_accuracy"],
        "loss": data["loss"],
        "val_loss": data["val_loss"],
    }

    fig_acc, ax_acc = plt.subplots(figsize=(12, 4))
    ax_acc.plot(history["accuracy"], label="Treino")
    ax_acc.plot(history["val_accuracy"], label="Validação")
    ax_acc.set_title("Acurácia do Modelo")
    ax_acc.set_xlabel("Época")
    ax_acc.set_ylabel("Acurácia")
    ax_acc.legend()

    fig_loss, ax_loss = plt.subplots(figsize=(12, 4))
    ax_loss.plot(history["loss"], label="Treino")
    ax_loss.plot(history["val_loss"], label="Validação")
    ax_loss.set_title("Loss do Modelo")
    ax_loss.set_xlabel("Época")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()

    return fig_acc, fig_loss


if __name__ == "__main__":
    main()
