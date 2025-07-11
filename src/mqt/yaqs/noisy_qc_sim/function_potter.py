import numpy as np
import matplotlib.pyplot as plt


def function_plotter(x):
    y = -0.5 * np.log(1-x*1.75)
    return y


if __name__ == "__main__":
    x = np.linspace(0, 0.2, 100)
    y = function_plotter(x)
    y_2 = x
    plt.plot(x, y)
    plt.plot(x, y_2)
    plt.show()