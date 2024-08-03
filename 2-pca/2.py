import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

scores = pd.read_csv('2-data1.csv', header=None, sep=';')
weights = pd.read_csv('2-data2.csv', header=None, sep=';')

# восстановление изображения по первым 10 главным компонентам
restored_image = np.dot(scores, weights.T)

# вывод картинки
plt.imshow(restored_image, cmap='Greens_r')
plt.show()
