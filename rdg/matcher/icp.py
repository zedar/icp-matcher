'''
Iterative Closest Point algorithm finds the best fitting transformation (rotation plus translation) between two point sets.
Uses internally SVD least squared best fit.sum
'''

import numpy as np
from sklearn.neighbors import NearestNeighbors

''' Calculate the least squares best fit transform that maps matrix x and y in m spatial dimensions
'''
def fit_point_sets(x, y):
  assert x.shape == y.shape

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
  if len(translation.shape) > 1:
    print('translation shape>1', translation)
    print('y_centroid.T', y_centroid.T)
    print('x_centroid.T', x_centroid.T)
  #print('translation.shape==1', translation)

  # homogeneus transformation
  t = np.identity(m+1)
  t[:m, :m] = r
  t[:m, m] = translation

  return t, r, translation 

''' Find the nearest (Euclidian) neighbour in dst for each point in src
'''
def nearest_neighbour(src, dst):
  assert src.shape == dst.shape

  neigh = NearestNeighbors(n_neighbors=1)
  neigh.fit(dst)
  distances, indices = neigh.kneighbors(src, return_distance=True)
  return distances.ravel(), indices.ravel()

''' ICP finds best fit transform that maps points a on to points y
'''
def icp(x, y, init_pose=None, max_iterations=20, error=0.01, tolerance=0.01):
  assert x.shape == y.shape
  
  # get dimensions
  m = x.shape[1]

  # make points homogeneous, copy them to maintain the originals
  # indexing array: (i:j:k) - i - from ith element, j - to jth element, k - with step
  # output arrays have last row (additional one) filled with ones
  src = np.ones((m+1, x.shape[0]))
  dst = np.ones((m+1, y.shape[0]))
  src[:m,:] = np.copy(x.T)
  dst[:m,:] = np.copy(y.T)
  print('src.shape, dst.shape',src.shape, dst.shape)

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

  # calculate final transformation
  x_src = np.ones((x.shape[0], x.shape[1]))
  x_src[:,:] = np.copy(x)
  t, _, _ = fit_point_sets(x_src, src[:m,:].T)

  return t, distances, i
