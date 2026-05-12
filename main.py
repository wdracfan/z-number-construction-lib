from z_number_construction import *

import numpy as np
import matplotlib.pyplot as plt

_, axes = plt.subplots(1, 2, figsize=(10, 5))
data = np.random.normal(2, 4, size=100)
z = ZNumber()
z.fit(data)
z.plot_over_data(axes[0], axes[1], data, color='red', lw=3)
plt.show()