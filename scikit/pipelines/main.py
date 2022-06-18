from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    iris = datasets.load_iris()

    X = iris.data[:, [2, 3]]
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=1, stratify=y)

    pipe = make_pipeline(StandardScaler(), PCA(n_components=2), LogisticRegression(random_state=1))

    pipe.fit(X_train, y_train)

    pipe.predict(X_test)

    print(pipe.score(X_test, y_test))
