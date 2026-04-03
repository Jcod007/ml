import csv
import numpy as np

def model(x, w, b):
    # Modele lineaire: y = w*x + b
    return w * x + b

with open("data.csv",mode="r") as file:
    reader = csv.reader(file)
    # Ignore la ligne d'entete: x,y
    next(reader)

    # Lit toutes les paires (x, y) du fichier CSV
    data = [(float(row[0]), float(row[1])) for row in reader]
    tab_x = np.array([row[0] for row in data], dtype=np.float64)
    tab_y = np.array([row[1] for row in data], dtype=np.float64)

# Normalisation pour stabiliser l'entrainement numerique
tab_x_mean = tab_x.mean()
tab_x_std = tab_x.std()
tab_y_mean = tab_y.mean()
tab_y_std = tab_y.std()

tab_x_norm = (tab_x - tab_x_mean) / tab_x_std
tab_y_norm = (tab_y - tab_y_mean) / tab_y_std

w_norm = 0.0
b_norm = 0.0
learning_rate = 0.01

# Descente de gradient sur les donnees normalisees
for _ in range(5000):
    y_pred = model(tab_x_norm, w_norm, b_norm)
    error = y_pred - tab_y_norm
    w_grad = (2 / len(tab_x_norm)) * np.sum(error * tab_x_norm)
    b_grad = (2/len(tab_x_norm)) * np.sum(error)
    w_norm -= learning_rate * w_grad
    b_norm -= learning_rate * b_grad

# Reconvertit les parametres vers l'echelle originale des donnees
w = (tab_y_std * w_norm) / tab_x_std
b = tab_y_mean + (tab_y_std * b_norm) - (w * tab_x_mean)

print(f"Final parameters: w={w}, b={b}")
print(f"Predicted value for x=7: {model(7, w, b)}")
