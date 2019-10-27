import numpy as np

import cv2
import matplotlib.pyplot as plt

import pysaliency
import pysaliency.plotting
from pysaliency.baseline_utils import BaselineModel

#mit1003_stimuli, mit1003_fixations = pysaliency.get_mit1003(location='test_datasets_py3')
#
#bandwidth = 0.0217
#regularization = 2.0e-13
#baseline = BaselineModel(mit1003_stimuli, mit1003_fixations,bandwidth=bandwidth, eps=regularization)
#
##image = np.random.randn(1024, 1024)
#image = np.random.randn(1800, 2900)
#log_density = baseline.log_density(image)
#
#pysaliency.plotting.visualize_distribution(log_density)
#
#np.save('centerbias_waldo.npy', log_density)


a = np.load('meta/centerbias.npy')
print(a.shape)
print(a)
#a = cv2.resize(a, (1280,1024), interpolation=cv2.INTER_CUBIC)
#np.save('centerbias_waldo.npy', a)

#myjet = np.array([[0.        , 0.        , 0.5       ],
#                  [0.        , 0.        , 0.99910873],
#                  [0.        , 0.37843137, 1.        ],
#                  [0.        , 0.83333333, 1.        ],
#                  [0.30044276, 1.        , 0.66729918],
#                  [0.66729918, 1.        , 0.30044276],
#                  [1.        , 0.90123457, 0.        ],
#                  [1.        , 0.48002905, 0.        ],
#                  [0.99910873, 0.07334786, 0.        ],
#                  [0.5       , 0.        , 0.        ]])
#
#heatmap = -np.log(-a + 0.0001)
#heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + .00001)
#heatmap = myjet[np.round(np.clip(heatmap*10, 0, 9)).astype('int'), :]
#heatmap = (heatmap*255).astype('uint8')
#cv2.imshow('heatmap', heatmap)
#cv2.imwrite('prior_heatmap.png', heatmap)
#cv2.waitKey(0)


#log = np.log(-a)
#
#plt.figure(1)
#m = plt.gca().matshow(-a, alpha=0.5, cmap=plt.cm.RdBu)
#plt.colorbar(m)
#plt.title('density prediction')
#plt.axis('off');
#plt.savefig('density.png')
#plt.close()

