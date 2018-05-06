'''
Plot data sets 
'''
import numpy as np
import matplotlib.pyplot as plt

'''
Plot point set 
'''
def plot_point_sets(xs, ys = None, aligned_ys=None):
  fig = plt.figure(1)
  ax1 = fig.add_subplot(211)
  ax2 = fig.add_subplot(212)
  ax1.plot(xs[:, 0],xs[:, 1], 'ro')
  if ys is not None:
    ax1.plot(ys[:, 0], ys[:,1], 'bo')

  if aligned_ys is not None:
    ax2.plot(xs[:, 0],xs[:, 1], 'ro')
    ax2.plot(aligned_ys[:, 0],aligned_ys[:, 1], 'bo')
    
  plt.axis('equal')
  plt.show()