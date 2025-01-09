
# Image Classifier

A deep learning model using Convolutional Neural Networks (CNN) to classify images of dogs and cats. The project features an interactive Streamlit interface for real-time image prediction and visualization of model training metrics.

## Features

- Upload images of dogs or cats for classification.
- Real-time predictions with confidence scores.
- Visualization of training history (accuracy, loss).
- Displays model architecture summary.

## Technologies Used

- **TensorFlow/Keras**: For the deep learning model.
- **Streamlit**: For building the web interface.
- **Plotly**: For interactive data visualizations.
- **Matplotlib**: For training history plots.

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/oEmanuelFirmino/image_classifier.git
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

## Model Training

The model is a Convolutional Neural Network (CNN) with the following architecture:
- 3 Convolutional layers with MaxPooling.
- Global Average Pooling layer.
- Fully connected Dense layer with 128 neurons.
- Dropout regularization to prevent overfitting.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
