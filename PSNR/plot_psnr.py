import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

name = sys.argv[0]

def plot_psnr(path_to_file, label):
  data = pd.read_csv(path_to_file, header=None)
  epochs = np.array(data[0])
  psnr = np.array(data[1])
  std = np.array(data[2])

  plt.errorbar(epochs, psnr, yerr=std, fmt='o', label=label)

print("Number of datasets: ", len(sys.argv)-1)
for i in range(1, len(sys.argv)):
  dataset = sys.argv[i]
  plot_psnr(dataset, dataset)

plt.xlabel("Number of Epochs")
plt.ylabel("PSNR")
plt.title("Peak Signal to Noise Ratio from Ground Truth")
plt.legend()
plt.savefig('psnr.png')
