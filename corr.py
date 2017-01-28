import matplotlib.pyplot as plt

def main():
    f = open('corr.txt')
    X_data = []
    for line in f:
        row = line.split(" ")
        row_new = []
        for i in range(len(row)):
            if row[i] != '':
                if i > 0 :
                    row_new.append((abs(float(row[i]))))
                else:
                    row_new.append(row[i])

        X_data.append(row_new)
    X_data.sort(key = lambda x: x[4])
    # # for x in X_data:
    # #     D[x[0]] = x[4]
    # corr = [x[4] for x in X_data]
    # label = [x[0] for x in X_data]
    # length = len(corr)
    # corr = corr[:length - 1]
    # label = label[:length - 1]
    #
    # plt.xlabel('Features')
    # plt.ylabel('Correlation')
    # plt.title('Correlation of features wrt to Target class')
    # plt.bar(range(len(corr)), corr, align='center')
    # plt.xticks(range(len(corr)), label, rotation='vertical')
    #
    # plt.show()

    for x in X_data:
        print(str(x[0]))




if __name__ == '__main__':
    # plot()
    main()