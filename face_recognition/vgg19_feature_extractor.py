import torch
from torchvision.models import vgg19


class VGG19FeatureExtractor:
    def __init__(self):
        model = vgg19(pretrained=True)
        # Use only the feature extractor part of VGG19
        self.feature_extractor = torch.nn.Sequential(*list(model.features.children())[:-1]).eval()

    def extract_features(self, img):
        with torch.no_grad():
            embedding = self.feature_extractor(img)
        return embedding.mean(dim=[2, 3])  # Global Average Pooling

    def __call__(self, img):
        return self.extract_features(img)
