import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

scores = pd.read_csv('https://courses.openedu.ru/assets/courseware/v1/453d2c47ea4f165db9546c68e00f24de/asset-v1:ITMOUniversity+MLDATAN+spring_2024_ITMO_bac+type@asset+block/X_reduced_792.csv', header=None, sep=';')
weights = pd.read_csv('https://courses.openedu.ru/assets/courseware/v1/78f586bc1239cec75ef777a7816d7b83/asset-v1:ITMOUniversity+MLDATAN+spring_2024_ITMO_bac+type@asset+block/X_loadings_792.csv', header=None, sep=';')

# восстановление изображения по первым 10 главным компонентам
restored_image = np.dot(scores, weights.T)

# вывод картинки
plt.imshow(restored_image, cmap='Greens_r')
plt.show()
