'''
Test matching on artifitially generated data
'''
import numpy as np

import rdg.matcher.data_generator as dg
import rdg.matcher.icp as icp
from rdg.utils.data_plot import plot_point_sets

def main():
  # target point set S
  S = dg.generate_data_sin(start=0, end=3, step=0.2)
  print('Input matrix S: ', type(S), ' shape:', S.shape)
  print('Input matrix S:\n', S)
  
  # reference point set M, to be aligned to S
  M = dg.generate_data_sin(start=1, end=2.8, step=0.2)
  t0, t1, k = dg.generate_transformations()
  M = dg.transform_data(M, t0, t1, k=1.0)
  print('Input matrix M: ', type(M), ' shape:', M.shape)
  print('Input matrix M:\n', M)

  # plot target point set S and point set M to align to S
  #plot_point_sets(S, M)
  
  # shuffle points in y matrix
  M_shuffled = dg.shuffle_data(M)

  t, distances, iterations, mean_error = icp.icp(M, S, max_iterations=2000, error=1e-2, tolerance=1e-9)

  print('Translation matrix:\n', t)

  aligned_ys = np.ones((M.shape[0], M.shape[1]+1))
  aligned_ys[:,0:M.shape[1]] = np.copy(M)
  aligned_ys = np.dot(t, aligned_ys.T)
  aligned_ys = aligned_ys.T
  aligned_ys = aligned_ys[:,:2]

  plot_point_sets(S, M, aligned_ys)

if __name__ == '__main__':
  main()