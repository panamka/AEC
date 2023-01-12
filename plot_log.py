import torch
import numpy as np
from matplotlib import pyplot as plt
import os


path = './results/last_snapshot.tar'
snapshot = torch.load(path, map_location='cpu')



if __name__ == '__main__':
    path = './results/conformer_causal'
    log = torch.load(os.path.join(path, 'last_snapshot.tar'), map_location=torch.device('cpu'))['val_history']

    print(path)

    n_plots = len(log)
    m = round(n_plots ** 0.5)
    n = n_plots // m
    if m * n < n_plots:
        n += 1
        # fig = plt.figure(figsize=(16, 9))
    fig = plt.figure()
    for i, (name, values) in enumerate(sorted(log.items())):
        ax = fig.add_subplot(m, n, i + 1)
        ax.set_title(name)
        print(name)
        plt.plot(values, label=name)

        ax.grid()
    plt.show()