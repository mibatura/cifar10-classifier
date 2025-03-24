
import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Define the CNN (same as in training)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# CIFAR-10 classes
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN()
model.load_state_dict(torch.load("simple_cnn_cifar10.pth", map_location=device))
model.to(device)
model.eval()

# Define image transform
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Streamlit UI
st.title("🧠 CIFAR-10 Image Classifier")
st.write("Upload an image and let the model predict the class!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output, 1)
        predicted_class = classes[predicted.item()]
    
    st.markdown(f"### 🔍 Prediction: **{predicted_class}**")
