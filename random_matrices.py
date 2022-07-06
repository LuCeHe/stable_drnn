import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
samples = 10
n = 10

maximals = []
for _ in tqdm(range(samples)):
    matrix = 2 * np.random.rand(n, n) - 1
    std = np.std(matrix)
    matrix /= std

    eigenvalues = np.linalg.eigvals(matrix)
    print(eigenvalues)
    print(abs(eigenvalues))
    x = eigenvalues.real
    y = eigenvalues.imag
    maximals.append(max(abs(eigenvalues)))
    plt.scatter(x, y)

print('mean vs sqrt n: ', np.mean(maximals), np.sqrt(n))