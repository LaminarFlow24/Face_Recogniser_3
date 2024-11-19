import os
import argparse
import joblib
import numpy as np
from PIL import Image
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
from torchvision import transforms, datasets
from face_recognition import preprocessing, VGG19FeatureExtractor, FaceRecogniser

MODEL_DIR_PATH = 'model'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train Face Recognition model using VGG19 for feature extraction and SVM for classification.'
    )
    parser.add_argument('-d', '--dataset-path', required=True, help='Path to folder with images.')
    parser.add_argument('--grid-search', action='store_true', help='Enable grid search for SVM hyperparameters.')
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

    return np.stack(embeddings), np.array(labels)


def train(args, X_train, y_train):
    svm = SVC(kernel='rbf', probability=True)  # Using RBF kernel for SVM
    if args.grid_search:
        clf = GridSearchCV(
            estimator=svm,
            param_grid={
                'C': [0.1, 1, 10, 100],  # Regularization parameter
                'gamma': [1e-3, 1e-4, 'scale', 'auto']  # Kernel coefficient
            },
            cv=3
        )
    else:
        clf = svm
    clf.fit(X_train, y_train)
    return clf.best_estimator_ if args.grid_search else clf


def main():
    args = parse_args()

    feature_extractor = VGG19FeatureExtractor()
    dataset = datasets.ImageFolder(args.dataset_path)
    embeddings, labels = dataset_to_embeddings(dataset, feature_extractor)
    class_to_idx = dataset.class_to_idx

    # Train-test split (80:20)
    X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, stratify=labels, random_state=42)

    clf = train(args, X_train, y_train)

    # Evaluate on the test set
    y_pred = clf.predict(X_test)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    target_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    print("Test Set Evaluation:")
    print(metrics.classification_report(y_test, y_pred, target_names=target_names))

    # Save the trained model
    if not os.path.isdir(MODEL_DIR_PATH):
        os.mkdir(MODEL_DIR_PATH)
    model_path = os.path.join(MODEL_DIR_PATH, 'face_recogniser.pkl')
    joblib.dump(FaceRecogniser(feature_extractor, clf, idx_to_class), model_path)


if __name__ == '__main__':
    main()