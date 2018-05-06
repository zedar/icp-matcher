'''
Test matching on artifitially generated data
'''
import numpy as np

import rdg.matcher.data_generator as dg
import rdg.matcher.icp as icp
from rdg.utils.data_plot import plot_point_sets

def main():
  x = dg.generate_data_sin()
  print('Input matrix X: ', type(x), ' shape:', x.shape)

  t0, t1, k = dg.generate_transformations()
  y = dg.transform_data(x, t0, t1, k=1.0)
  print('Input matrix Y: ', type(y), ' shape:', y.shape)

  # shuffle points in y matrix
  ys = dg.shuffle_data(y)

  t, distances, iterations = icp.icp(ys, x, max_iterations=1500, error=1e-2, tolerance=1e-9)

  print('Iterations:', iterations)
  print('Translation matrix:\n', t)

  aligned_ys = np.ones((y.shape[0], y.shape[1]+1))
  aligned_ys[:,0:y.shape[1]] = np.copy(ys)
  aligned_ys = np.dot(t, aligned_ys.T)
  aligned_ys = aligned_ys.T
  aligned_ys = aligned_ys[:,:2]
  #print('Aligned ys:\n', aligned_ys) 
  #print('xs:\n', x)

  plot_point_sets(x, y, aligned_ys)

if __name__ == '__main__':
  main()