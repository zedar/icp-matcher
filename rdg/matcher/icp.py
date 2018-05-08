'''
Iterative Closest Point algorithm finds the best fitting transformation (rotation plus translation) between two point sets.
Uses internally SVD least squared best fit.sum
'''

import numpy as np
from sklearn.neighbors import NearestNeighbors

''' Calculate the least squares best fit transform that maps matrix x and y in m spatial dimensions
'''
def fit_point_sets(x, y):
  assert x.shape[0] <= y.shape[0]
  assert x.shape[1] == y.shape[1]

  # dimension of points
  m = x.shape[1]

  # translate points to their centroids
  x_centroid = np.mean(x, axis=0)
  y_centroid = np.mean(y, axis=0)
  xx = x - x_centroid
  yy = y - y_centroid

  # rotation matrix
  h = np.dot(xx.T, yy)
  u, s, vt = np.linalg.svd(h)
  r = np.dot(vt.T, u.T)

  # reflection case
  if np.linalg.det(r) < 0:
    vt[m-1,:] *= -1
    r = np.dot(vt.T, u.T)

  # translation
  translation = y_centroid.T - np.dot(r, x_centroid.T)

  # homogeneus transformation
  t = np.identity(m+1)
  t[:m, :m] = r
  t[:m, m] = translation

  return t, r, translation 

''' Find the nearest (Euclidian) neighbour in dst for each point in src
'''
def nearest_neighbour(src, dst):
  assert src.shape[0] <= dst.shape[0]
  assert src.shape[1] == dst.shape[1]

  neigh = NearestNeighbors(n_neighbors=1)
  neigh.fit(dst)
  distances, indices = neigh.kneighbors(src, return_distance=True)
  return distances.ravel(), indices.ravel()

"""
ICP finds best fit transform that maps points X on to points Y.
Y matrix is target (fixed set of features), while X is a set to align to Y.
Args:
  X: matrix of points to align to Y
  Y: matrix of target points
  init_pose: the initial transformation
  max_iterations: to stop iterations at
  tolerance: convergence criteria
Returns:
  T: final transformation matrix
  distances: distance for every point to the neerest neighbour
  i: number of iterations
  mean_error: mean error for the last iteration
"""
def icp(x, y, init_pose=None, max_iterations=20, error=0.01, tolerance=0.01):
  assert x.shape[0] <= y.shape[0]
  assert x.shape[1] == y.shape[1]
  
  # number of features describing every point in both, target (Y) and reference (X) point sets
  m = x.shape[1]

  # make points homogeneous, copy them to maintain the originals
  # indexing array: (i:j:k) - i - from ith element, j - to jth element, k - with step
  # output arrays have last row (additional one) filled with ones
  src = np.ones((m+1, x.shape[0]))
  dst = np.ones((m+1, y.shape[0]))
  src[:m,:] = np.copy(x.T)
  dst[:m,:] = np.copy(y.T)

  # initial pose position
  if init_pose is not None:
    src = np.dot(init_pose, src)
  
  prev_error = 0

  for i in range(max_iterations):
    # find the nearest neighbours bewteen the current source and destination points
    distances, indices = nearest_neighbour(src[:m,:].T, dst[:m,:].T)

    # compute the transformation between the current source and nearest destination points
    t, _, _ = fit_point_sets(src[:m,:].T, dst[:m, indices].T)

    # update the current source
    src = np.dot(t, src)

    # calculate error
    mean_error = np.mean(distances)
    #print('Mean Error:', mean_error, 'Tolerance:', np.abs(prev_error-mean_error))
    # if np.abs(prev_error - mean_error) < tolerance:
    #   break
    if mean_error < error:
      break
    prev_error = mean_error
  
  print('Mean error:', mean_error, ' after iteration:', i)
  
  # calculate final transformation
  x_src = np.copy(x)
  t, _, _ = fit_point_sets(x_src, src[:m,:].T)

  return t, distances, i, mean_error

def icp2(x, y, init_pose=None, max_iterations=20, error=0.01, tolerance=0.01):
  assert x.shape[0] <= y.shape[0]
  assert x.shape[1] == y.shape[1]
  
  # number of features describing every point in both, target (Y) and reference (X) point sets
  m = x.shape[1]
  dims = 3

  # make points homogeneous, copy them to maintain the originals
  # indexing array: (i:j:k) - i - from ith element, j - to jth element, k - with step
  # output arrays have last row (additional one) filled with ones
  src_xyz = np.ones((dims+1, x.shape[0]))
  dst_xyz = np.ones((dims+1, x.shape[0]))
  src = np.copy(x.T)
  dst = np.copy(y.T)
  #dst = np.ones((m+1, y.shape[0]))
  #src[:m,:] = np.copy(x.T)
  #dst[:m,:] = np.copy(y.T)

  # initial pose position
  if init_pose is not None:
    src = np.dot(init_pose, src)
  
  prev_error = 0

  for i in range(max_iterations):
    # find the nearest neighbours bewteen the current source and destination points
    #distances, indices = nearest_neighbour(src[:m,:].T, dst[:m,:].T)
    distances, indices = nearest_neighbour(src[:m,:].T, dst[:m,:].T)

    # compute the transformation between the current source and nearest destination points
    #t, _, _ = fit_point_sets(src[:m,:].T, dst[:m, indices].T)
    t, _, _ = fit_point_sets(src[:dims,:].T, dst[:dims, indices].T)

    # update the current source
    #src = np.dot(t, src)
    src_xyz = np.dot(t, src_xyz)
    src[:dims,:] = src_xyz[:dims,:]

    # calculate error
    mean_error = np.mean(distances)
    #print('Mean Error:', mean_error, 'Tolerance:', np.abs(prev_error-mean_error))
    # if np.abs(prev_error - mean_error) < tolerance:
    #   break
    if mean_error < error:
      break
    prev_error = mean_error
  
  print('Mean error:', mean_error, ' after iteration:', i)
  
  # calculate final transformation
  x_src = np.copy(x[:, :dims])
  t, _, _ = fit_point_sets(x_src, src[:dims,:].T)

  return t, distances, i, mean_error

