import numpy as np

combinations = np.zeros(13)
for i in range(1, 7):
    for j in range (1, 7):
        combinations[i+j] += 1


single_probability = combinations / 6 / 6# * players * rounds
players = 5
rounds = 12
probability = combinations / 6 / 6 * players * rounds

print(single_probability * 100)
