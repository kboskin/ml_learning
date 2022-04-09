# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from adaline import AdalineGraientPacket, AdalineGradientStohastic
from perceptron import Perceptron

from sklearn import datasets


# See PyCharm help at https://www.jetbrains.com/help/pycharm/

def plot_decision_regions(X, y, classifier, resolution=0.02):
    markers = ('s', 'x', 'o', '4', '4')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    z = z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(
            x=X[y == cl, 0],
            y=X[y == cl, 1],
            alpha=0.8,
            c=colors[idx],
            marker=markers[idx],
            label=cl,
            edgecolors='black'
        )


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    s = os.path.join('../iris.data')
    print('URL', s)

    df = pd.read_csv(s, header=None, encoding='utf-8')
    y = df.iloc[0: 100, 4].values

    y = np.where(y == 'Iris-setosa', -1, 1)
    X = df.iloc[0:100, [0, 2]].values

    plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='scehtinisty')
    plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='raznotsvetny')

    plt.xlabel('lenght of chashelistniq')
    plt.ylabel('lenght of lepestoc')
    plt.legend(loc='upper left')
    plt.show()

    ppn = Perceptron(eta=0.1, n_iter=10)
    ppn.fit(X, y)

    plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
    plt.xlabel('epochs')
    plt.ylabel('Num of updates')
    plt.show()

    plot_decision_regions(X, y, ppn)

    plt.xlabel('lenght of chashelistniq')
    plt.ylabel('lenght of lepestoc')
    plt.legend(loc='upper left')
    plt.show()

    # gradient method (packet), without standartization
    ada1 = AdalineGraientPacket(eta=0.01, n_iter=10)
    ada1.fit(X, y)

    ada2 = AdalineGraientPacket(eta=0.0001, n_iter=10)
    ada2.fit(X, y)

    fi, ax = plt.subplots(nrows=1, ncols=2)

    ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
    ax[1].plot(range(1, len(ada2.cost_) + 1), np.log10(ada2.cost_), marker='o')

    plt.show()

    # with standartization

    X_std = np.copy(X)

    X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
    X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

    ada3 = AdalineGraientPacket(n_iter=15, eta=0.01, random_state=1)
    ada3.fit(X_std, y)

    plot_decision_regions(X_std, y, classifier=ada3)

    plt.title('Adaline - gradient down')
    plt.xlabel('lenght of chashelistniq')
    plt.ylabel('lenght of lepestoc')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

    plt.plot(range(1, len(ada3.cost_) + 1), ada3.cost_, marker='o')
    plt.xlabel('Epochs')
    plt.xlabel('Sum of square errors')
    plt.tight_layout()
    plt.show()

    # gradient method (stohastic), without standartization

    ada4 = AdalineGradientStohastic(eta=0.01, n_iter=15, random_state=1)
    ada4.fit(X_std, y)

    plot_decision_regions(X_std, y, classifier=ada4)

    plt.title('Adaline stohastic - gradient down')
    plt.xlabel('lenght of chashelistniq')
    plt.ylabel('lenght of lepestoc')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

    plt.plot(range(1, len(ada4.cost_) + 1), ada4.cost_, marker='o')
    plt.xlabel('Epochs')
    plt.xlabel('Sum of square errors')
    plt.tight_layout()
    plt.show()
