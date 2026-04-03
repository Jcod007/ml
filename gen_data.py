import csv
import numpy as np

def unknown_function(x):
    return 10 * x +10

tab_x = np.arange(-2000,2000, 1)
tab_y = np.zeros(len(tab_x))

if __name__ == "__main__":
    for i, x in enumerate(tab_x):
        tab_y[i] = unknown_function(x)
    print(tab_y)

    with open("data.csv", mode = "w", newline ="") as  file:
        writer = csv.writer(file)
        writer.writerow(["x","y"])
        writer.writerows(zip(tab_x, tab_y))
