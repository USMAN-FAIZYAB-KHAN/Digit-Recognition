# 🖋️ **Digit Recognition System**

Welcome to the **Digit Recognition System**, a Django-based web application designed to recognize handwritten digits. This application uses a custom-trained neural network model to accurately predict digits drawn by the user on an interactive canvas.

---

## 📜 **Table of Contents**

1. [Overview](#-overview)
2. [Features](#-features)
3. [Screenshots](#-screenshots)
4. [Getting Started](#-getting-started)
5. [Model Details](#-model-details)
6. [Dependencies](#-dependencies)
7. [Contributors](#-contributors)

---

## 📜 **Overview**

The **Digit Recognition System** enables users to draw digits on an interactive canvas and receive real-time predictions powered by a custom-trained neural network. This project showcases an end-to-end pipeline, from training the model using the MNIST dataset to deploying the model in a web interface built with Django.

---

## ✨ **Features**

### - **Custom Neural Network** 🧠
  - Implemented and trained from scratch using **TensorFlow**.
  - Built on the **MNIST** dataset, which consists of grayscale 28x28 images of handwritten digits.
  - Offers high accuracy for recognizing handwritten digits.

### - **Digit Drawing Canvas** 🖌️
  - Interactive web interface allowing users to draw digits for immediate prediction.
  - Uses HTML5 canvas for easy digit input and drawing.
  
### - **End-to-End Deployment** 🌐
  - Combines all steps, including data preprocessing, training, prediction, and web deployment, within a single system.
  - Seamless integration of the trained model with the Django backend and front-end components.

### - **Real-Time Predictions** 🚀
  - The system predicts drawn digits as soon as the user finishes drawing, providing instant feedback.
  - Uses the trained neural network for inference directly in the web application.

### - **Neural Network Visualization** 🔍
  - Visualize the internal workings of the neural network and its training progress.
  - Display metrics such as accuracy and loss during training.
  - Shows how the model's weights evolve over time, offering insight into the decision-making process of the neural network.
  - Can be accessed through the web interface for an interactive learning experience.

---

## 📸 **Screenshots**

### - **Canvas Drawing**
![Canvas Drawing](./Screenshots/drawing_canvas.png)

### - **Prediction Output**
![Prediction Output](./Screenshots/prediction_result.png)

### - **Neural Network Visualization**
![Neural Network Visualization](./Screenshots/nn_visualization.png)

---

## 🚀 **Getting Started**

### 1. Clone the Repository:

```bash
git clone https://github.com/USMAN-FAIZYAB-KHAN/Digit-Recognition.git
cd Digit-Recognition
```

### 2. Install Dependencies:

Ensure **Python** and **pip** are installed. Then, install the required dependencies using pip:

```bash
pip install django numpy tensorflow
```

### 3. **Train the Model (Optional):**

If you'd like to retrain the model with updated settings or hyperparameters:

```bash
python neural_network.py
```

The model will be trained using the MNIST dataset, and a saved **pickle** file (`digit_recognition_model.pkl`) will be generated for recognition. This pickle file stores the model's weights and biases, allowing for easy loading and use in predictions.

### 4. Run the Server:

Start the Django development server:

```bash
python manage.py runserver
```

### 5. Access the Application:

Once the server is running, open your web browser and navigate to:

```
http://127.0.0.1:8000
```

---

## 🗝 **Technical Details**


---

## 📦 **Dependencies**

The project relies on the following dependencies:

- **Django**: Web framework for the backend.
- **NumPy**: For numerical processing.
- **TensorFlow**: Used only for the MNIST dataset, not for a pre-trained model.
- **Pickle**: Built-in python library for saving and loading the model.

---

## 🤝 **Contributors**

- [**Usman Faizyab Khan**](https://github.com/USMAN-FAIZYAB-KHAN)
- [**Muhammad Owais**](https://github.com/MuhammadOwais03)
- [**Muhammad Zunain**](https://github.com/Muhammad-Zunain)
- [**Zuhaib Noor**](https://github.com/zuhaibnoor)

---

Enjoy exploring the **Digit Detection System**! Contributions and feedback are welcome. 😊
