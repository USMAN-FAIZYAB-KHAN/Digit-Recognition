from django.shortcuts import render
from django.views.decorators.http import require_POST
from django.http import JsonResponse
import base64
import json
import io
from PIL import Image
import numpy as np
import pickle
from .neural_network import predict_image, LEARNING_RATE, LAYERS, ERROR_THRESHOLD

# Load weights and biases
with open('./digit_detection_app/digit_recognition_model.pkl', 'rb') as f:
    data = pickle.load(f)
    weights = data['weights']
    biases = data['biases']

# Create your views here.
def index(request):
    return render(request, 'index.html')

@require_POST
def predict(request):
    binary_image = request.POST.get('image')
    binary_image = binary_image.split(",")[1]
    binary_image = base64.b64decode(binary_image)

    image = Image.open(io.BytesIO(binary_image)).resize((28, 28)).convert('L')
    image_array = np.array(image)

    image_array = image_array / 255.0
    image_array = image_array.reshape(1 , 784)  # Shape (784, 1)

    predictions = predict_image(image_array, weights, biases)
    prediction = np.argmax(predictions)
    percentages = (predictions[0] * 100).round(2)

    return JsonResponse({'predictedDigit': str(prediction), "status": "success", "predictions": percentages.tolist()})

def neural_network(request):
    return render(request, 'neural_network.html', {'layers': LAYERS})

def check(request):
    return render(request, 'check.html')

def train(request):
    return render(request, 'train.html')

def graph(request):
    error_history = json.load(open('./digit_detection_app/error_history.json', 'r'))
    return render(request, 'graph.html', { "error_history": error_history })

def parameters(request):
    return render(request, 'parameters.html', { "layers": LAYERS, "learning_rate": LEARNING_RATE, "error_threshold": ERROR_THRESHOLD })
