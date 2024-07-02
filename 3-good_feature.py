import numpy as np
import matplotlib.pyplot as plt

greyhounds = 500
labs = 500

grey_height = 28 + 4*np.random.randn(greyhounds)
lab_height = 24 + 4*np.random.randn(labs)

plt.hist([grey_height, lab_height], stacked=True, color=['r', 'b'])
plt.show()

'''
These features are somewhat useful.
But, features like eye-colour do not help much in predicting the breed of the dog
Also, height_inches and height_cm are redundant, so we do not want highly correlated features
'''