import sys
import os

CNNDETECTION_REPO_PATH = r"C:\Users\Administrador\Documents\Projetos\CNNDetection"
sys.path.insert(0, CNNDETECTION_REPO_PATH)

import torch
import torchvision.transforms as transforms
from PIL import Image
from networks.resnet import resnet50

class CNNDetector:
    def __init__(self, weights_path=None, use_gpu=True):
        if weights_path is None:
            # Pasta "weights" que está do lado deste arquivo detection.py
            weights_path = os.path.join(os.path.dirname(__file__), 'weights', 'blur_jpg_prob0.5.pth')

        self.model = resnet50(num_classes=1)
        state_dict = torch.load(weights_path, map_location='cpu')
        self.model.load_state_dict(state_dict['model'])
        self.use_gpu = use_gpu and torch.cuda.is_available()
        if self.use_gpu:
            self.model.cuda()
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def predict(self, image_path: str) -> float:
        img = Image.open(image_path).convert('RGB')
        tensor = self.transform(img).unsqueeze(0)
        if self.use_gpu:
            tensor = tensor.cuda()
        with torch.no_grad():
            prob = self.model(tensor).sigmoid().item()
        return prob