from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def knn_classification(pca_features, daisy_features, labels, n_neighbors=1):
    pca_train, pca_test, dsy_train, dsy_test, labels_train, labels_test = train_test_split(
        pca_features, daisy_features, labels, test_size=0.2, train_size=0.8)

    knn_pca = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_dsy = KNeighborsClassifier(n_neighbors=n_neighbors)

    knn_pca.fit(pca_train, labels_train)
    knn_dsy.fit(dsy_train, labels_train)

    acc_pca = accuracy_score(knn_pca.predict(pca_test), labels_test)
    acc_dsy = accuracy_score(knn_dsy.predict(dsy_test), labels_test)

    return acc_pca, acc_dsy
