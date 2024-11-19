import os
import argparse
import joblib
import numpy as np
from PIL import Image
from torchvision import transforms, datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from face_recognition import preprocessing, VGG19FeatureExtractor, FaceRecogniser

MODEL_DIR_PATH = 'model'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train Face Recognition model using VGG19 for feature extraction.'
    )
    parser.add_argument('-d', '--dataset-path', help='Path to folder with images.')
    parser.add_argument('-e', '--embeddings-path', help='Path to file with embeddings.')
    parser.add_argument('-l', '--labels-path', help='Path to file with labels.')
    parser.add_argument('-c', '--class-to-idx-path', help='Path to pickled class_to_idx dict.')
    parser.add_argument('--grid-search', action='store_true', help='Enable grid search for Logistic Regression.')
    return parser.parse_args()


def dataset_to_embeddings(dataset, feature_extractor):
    transform = transforms.Compose([
        preprocessing.ExifOrientationNormalize(),
        transforms.Resize((224, 224)),  # Input size for VGG19
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # VGG19 normalization
    ])
    embeddings = []
    labels = []
    for img_path, label in dataset.samples:
        print(f"Processing: {img_path}")
        img = transform(Image.open(img_path).convert('RGB'))
        embedding = feature_extractor(img.unsqueeze(0))  # Add batch dimension
        embeddings.append(embedding.flatten().detach().numpy())
        labels.append(label)

    return np.stack(embeddings), labels


def load_data(args, feature_extractor):
    if args.embeddings_path:
        return np.loadtxt(args.embeddings_path), \
               np.loadtxt(args.labels_path, dtype='str').tolist(), \
               joblib.load(args.class_to_idx_path)

    dataset = datasets.ImageFolder(args.dataset_path)
    embeddings, labels = dataset_to_embeddings(dataset, feature_extractor)
    return embeddings, labels, dataset.class_to_idx


def train(args, embeddings, labels):
    softmax = LogisticRegression(solver='lbfgs', multi_class='multinomial', C=10, max_iter=10000)
    if args.grid_search:
        clf = GridSearchCV(
            estimator=softmax,
            param_grid={'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]},
            cv=3
        )
    else:
        clf = softmax
    clf.fit(embeddings, labels)
    return clf.best_estimator_ if args.grid_search else clf


def main():
    args = parse_args()

    feature_extractor = VGG19FeatureExtractor()
    embeddings, labels, class_to_idx = load_data(args, feature_extractor)
    clf = train(args, embeddings, labels)

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    target_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    print(metrics.classification_report(labels, clf.predict(embeddings), target_names=target_names))

    if not os.path.isdir(MODEL_DIR_PATH):
        os.mkdir(MODEL_DIR_PATH)
    model_path = os.path.join(MODEL_DIR_PATH, 'face_recogniser.pkl')
    joblib.dump(FaceRecogniser(feature_extractor, clf, idx_to_class), model_path)


if __name__ == '__main__':
    main()
