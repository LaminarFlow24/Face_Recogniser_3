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



import cv2


def augment_image(image):
    """Apply a series of augmentations to the image to generate 10 variations."""
    augmented_images = []

    # 1. Flip (Horizontal)
    flipped = cv2.flip(image, 1)
    augmented_images.append(flipped)

    # 2. Rotate (15 degrees)
    h, w = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((w // 2, h // 2), 15, 1)
    rotated_15 = cv2.warpAffine(image, rotation_matrix, (w, h))
    augmented_images.append(rotated_15)

    # 3. Rotate (-15 degrees)
    rotation_matrix = cv2.getRotationMatrix2D((w // 2, h // 2), -15, 1)
    rotated_minus_15 = cv2.warpAffine(image, rotation_matrix, (w, h))
    augmented_images.append(rotated_minus_15)

    # 4. Brightness Adjustment (Increase)
    brighter = cv2.convertScaleAbs(image, alpha=1.2, beta=30)
    augmented_images.append(brighter)

    # 5. Brightness Adjustment (Decrease)
    darker = cv2.convertScaleAbs(image, alpha=0.8, beta=-30)
    augmented_images.append(darker)

    # 6. Gaussian Blur
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    augmented_images.append(blurred)

    # 7. Random Crop
    start_x = int(w * 0.1)
    start_y = int(h * 0.1)
    cropped = image[start_y:h - start_y, start_x:w - start_x]
    resized_crop = cv2.resize(cropped, (w, h))  # Resize back to original size
    augmented_images.append(resized_crop)

    # 8. Horizontal Shift
    translation_matrix = np.float32([[1, 0, 20], [0, 1, 0]])  # Shift by 20 pixels horizontally
    shifted = cv2.warpAffine(image, translation_matrix, (w, h))
    augmented_images.append(shifted)

    # 9. Add Noise
    noise = np.random.normal(0, 25, image.shape).astype(np.uint8)  # Gaussian noise
    noisy_image = cv2.add(image, noise)
    augmented_images.append(noisy_image)

    # 10. Scale (Zoom-In)
    scale_matrix = cv2.getRotationMatrix2D((w // 2, h // 2), 0, 1.2)  # Scale up by 20%
    scaled = cv2.warpAffine(image, scale_matrix, (w, h))
    augmented_images.append(scaled)

    return augmented_images[:10]  # Ensure exactly 10 images are returned




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
        
        # Load image and convert to array
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

        # Apply augmentations
        augmented_images = [img] + augment_image(img)

        for aug_img in augmented_images:
            pil_img = Image.fromarray(aug_img)  # Convert to PIL image for transformations
            transformed_img = transform(pil_img)
            embedding = feature_extractor(transformed_img.unsqueeze(0))  # Add batch dimension
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