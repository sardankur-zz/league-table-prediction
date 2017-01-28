import matplotlib.pyplot as plt
from Classifier_top_features import select_features

def main():
    features = ["rating_home",
                "5HD",
                "5AD",
                "H2H5HR",
                "H2H5AR",
                "5AR",
                "5HR",
                "5FTAGC",
                "5AL",
                "5AC",
                "rating_away",
                "H2H5AC",
                "H2H5AST",
                "5AW",
                "5FTAGS",
                "H2H5HC",
                "5AST",
                "5FTHGC",
                "H2H5HST",
                "5HC",
                "H2H5FTAGS",
                "H2H5FTHGC",
                "5HL",
                "5HST",
                "5HW",
                "5FTHGS",
                "H2H5FTHGS",
                "H2H5FTAGC"]

    Y = []
    X = []
    z = 0
    iters = 10
    for k in range(iters):
        if k == 0:
            for i in range(5, len(features), 2):
                X.append(i)
                Y.append(select_features(features[len(features) - i:len(features)]))
        else:
            z = 0
            for i in range(5, len(features), 2):
                Y_n = select_features(features[len(features) - i:len(features)])
                for j in range(len(Y_n)):
                    Y[z][j] += Y_n[j]
                z += 1


    for y in Y:
        for i in range(len(y)):
            y[i] = y[i]/iters
    Y = [[y[0], y[1], y[2], y[3]] for y in Y]
    plt.plot(X, Y)

    classifiers_names = ['SVM', 'LinearSVM', 'Random Forest', 'Linear Regression']

    plt.xlabel('Top features used')
    plt.ylabel('Accuracy')
    plt.title('Feature Selection')
    plt.grid(True)
    plt.legend(classifiers_names)
    plt.show()



if __name__ == '__main__':
    main()