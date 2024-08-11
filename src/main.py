from src.data_preparation import load_and_resize_image
from src.pca_analysis import perform_pca, plot_explained_variance
from src.daisy_features import extract_daisy_features
from src.knn_classification import knn_classification


def main():
    folder_path = "./data/raw/training"
    target_size = (512, 512)
    num_images = 1500

    images, labels = load_and_resize_image(
        folder_path, target_size, num_images)
    h, w = 512, 512

    # Perform PCA
    pca = perform_pca(images, 300)
    plot_explained_variance(pca, "PCA Explained Variance")

    # Extract DAISY features
    daisy_features = extract_daisy_features(images, h, w)

    # KNN Classification
    acc_pca, acc_daisy = knn_classification(
        pca.transform(images), daisy_features, labels)

    print(f"PCA accuracy: {acc_pca*100:.2f}%")
    print(f"DAISY accuracy: {acc_daisy*100:.2f}%")


if __name__ == "__main__":
    main()
