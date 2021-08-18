import matplotlib.pyplot as plt
import numpy as np

def plot_learning_curve():

    with open("..//learning_curves//adaboost_learning_data.txt", mode='r') as file:
        data = file.read().splitlines()
        list_data = []
        for d in data:
            d = d.split(',')
            list_data.append(d)
        data = np.array(list_data[1:]).astype(np.float)

    fig = plt.figure(figsize=(10, 6))

    fig.suptitle("AdaBoost Learning Curves")

    acc_chart = fig.add_subplot(121)
    acc_chart.plot(data[1:, 0], data[1:, 1], color="k")
    acc_chart.plot(data[1:, 0], data[1:, 2], color="r")
    acc_chart.set_xlabel("Percentage of data used.")
    acc_chart.set_ylabel("Accuracy")
    acc_chart.legend(["Train Data", "Test Data"])

    error_chart = fig.add_subplot(122)
    error_chart.plot(data[1:, 0], data[1:, 3], color="k")
    error_chart.plot(data[1:, 0], data[1:, 4], color="r")
    error_chart.set_xlabel("Percentage of data used.")
    error_chart.set_ylabel("Error")
    error_chart.legend(["Train Data", "Test Data"])

    plt.show()
