'''
Test matching on artifitially generated data
'''
import numpy as np

import rdg.matcher.data_generator as dg
import rdg.matcher.icp as icp
from rdg.utils.data_plot import plot_point_sets

def main():
  X = dg.generate_data_sin(start=0, end=3, step=0.2)
  print('Input matrix X: ', type(X), ' shape:', X.shape)
  
  Y = dg.generate_data_sin(start=1, end=2.8, step=0.2)
  t0, t1, k = dg.generate_transformations()
  Y = dg.transform_data(Y, t0, t1, k=1.0)
  print('Input matrix Y: ', type(Y), ' shape:', Y.shape)

  plot_point_sets(X, Y)
  
  # shuffle points in y matrix
  Y_shuffled = dg.shuffle_data(Y)

  t, distances, iterations = icp.icp(Y, X, max_iterations=1500, error=1e-2, tolerance=1e-9)

  print('Iterations:', iterations)
  print('Translation matrix:\n', t)

  aligned_ys = np.ones((Y.shape[0], Y.shape[1]+1))
  aligned_ys[:,0:Y.shape[1]] = np.copy(Y)
  aligned_ys = np.dot(t, aligned_ys.T)
  aligned_ys = aligned_ys.T
  aligned_ys = aligned_ys[:,:2]
  #print('Aligned ys:\n', aligned_ys) 
  #print('xs:\n', x)

  plot_point_sets(X, Y, aligned_ys)

if __name__ == '__main__':
  main()