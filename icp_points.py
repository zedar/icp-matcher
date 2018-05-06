'''
Iterative closest point algorithm (ICP).
It seeks to find a transformation between two sets of points that minimizes the error between them.

This example assumes that one set of points is a rotated, shifted and scaled version of the set of target points.
'''

import numpy as np
import scipy.stats
from pylab import *
import random

''' Generate random rotation and translation matrix as well as scaling factor
:return: tuple of rotation matrix, translation matrix, scaling factor
'''
def generate_transformations():
  # rotate sample data by angle theta, thats value is between 0 and 2PI
  theta = scipy.stats.uniform(0.0, 2.0*np.pi).rvs()
  print('Rotation angle rad:', theta, ' degs: ', np.rad2deg(theta))

  # calculate rotation matrix
  c, s = np.cos(theta), np.sin(theta)
  t0 = np.matrix([[c,-s], [s,c]])
  print('Rotation matrix: \n', t0)

  # calculate translation vector
  t1 = np.matrix(scipy.stats.norm(loc=3, scale=3).rvs(size=(3,2)))
  print('Translation vector: \n', t1)

  # calculate scaling factor
  k = scipy.stats.uniform(1, 5).rvs()
  print('Scaling factor: ', k)

  return (t0, t1, k)

''' Generate matrix of sample points
:return: matrix od data
'''
def generate_data():
  # generate random numbers (rvs, in location 500 and scale 2) from normal distribution
  samples = scipy.stats.norm(loc=0, scale=100).rvs(size=(100,2))
  x = np.matrix(samples)
  return x

''' Transform sample data by rotation, translation and scaling
:param matrix x: matrix of data to transform
:param t0: rotation matrix
:param t1: transformation matrix
:param k: scaling factor
:return: matrix of transformed data
'''
def transform_data(x, t0, t1, k=1.0):
  # apply an affine transformation to x
  y = x * t0.T
  #print('x shape: ', x.shape)
  #print('y shape: ', y.shape)
  #print('t1 shape: ', t1.shape)
  y[:,0] += t1[0,0]
  y[:,1] += t1[1,0]
  return y*k

''' Shufle some points in matrix randomly
:param x: matrix to shuple same data
:return: a matrix with shufled data
'''
def shuffle_data(x):
  # get number of rows of matrix x
  n = x.shape[0]
  # get list of o to number of rows
  idx = list(range(n))
  # shuffle in place indices
  np.random.shuffle(idx)
  # return matrix with shufled indices
  return x[idx]

''' Translate to align the centerd of mass
:param x: matrix x
:param y: transformed matrix y, that we try to align with matrix x
:param error_fn: error function
'''
def translate(x, y, error_fn):
  mx = np.mean(x, axis=0).T
  my = np.mean(y, axis=0).T
  #print('mx shape: ', mx.shape, 'my shape', my.shape)
  translation = mx - my
  # get identity matrix
  i = np.matrix(np.eye(2))
  yp = transform_data(y, i, translation)
  return error_fn(x, yp), translation

''' Do random rotation
:param x: matrix x
:param y: transformed matrix y, that we try to align with matrix x
:param error_fn: error function
'''
def random_rotation(x, y, error_fn):
  theta = scipy.stats.uniform(0.0, 2.0*np.pi).rvs()
  c, s = np.cos(theta), np.sin(theta)
  rotation = np.matrix([[c, -s], [s,c]])
  z = np.matrix(np.zeros((2,1)))
  yp = transform_data(y, rotation, z)
  return error_fn(x, yp), rotation

''' Scale by random coeficient
'''
def random_scale(x, y, error_fn):
  k = scipy.stats.uniform(0.5, 1.0).rvs()
  scaling = k * np.matrix(np.eye(2))
  z = np.matrix(np.zeros((2,1)))
  yp = transform_data(y, scaling, z)
  return error_fn(x, yp), scaling

''' Sum squared errors
'''
def sum_squared_error(x, y):
  return np.sum(np.array(x-y)**2.0)

''' Point-wise smallest squared error. This is the distance from point pt to the closest point in x
'''
def point_sum_squared_error(pt, x):
  difference = pt - x
  xcol = np.ravel(difference[:,0])
  ycol = np.ravel(difference[:,1])
  # sum the squared differences between pt and x
  sqrdiff = xcol**2.0 + ycol**2.0
  distance = np.min(sqrdiff)
  # index of the nearest point to pt in x
  nearest_pt = np.argmin(sqrdiff)
  return distance

''' Nearest sum squered error
'''
def nearest_squared_error(x, y):
  err = 0.0
  for xi in x:
    err += point_sum_squared_error(xi, y)
  return err

''' Fit one point set to another one
'''
def fit_point_sets(x, y, m, n, error_fn, threshold=1e-5):
  t0 = list()
  t1 = list()
  errors = list()
  errors.append(error_fn(x, y))
  print(errors[-1])
  yp = y.copy()
  for iter in range(m):
    err, translation = translate(x, yp, error_fn)
    if err < threshold:
      print(str(iter).zfill(3), ": ", err)
      break
    elif err < errors[-1]:
      errors.append(err)
      print(str(iter).zfill(3), ": ", errors[-1])
      t1.append(translation)
      i = np.matrix(np.eye(2))
      yp = transform_data(yp, i, t1[-1])

    rot = [random_rotation(x, yp, error_fn) for i in range(n)]
    rot.sort()
    err, rotation = rot[0]
    if err < threshold:
      print(str(iter).zfill(3), ": ", err)
      break
    elif err < errors[-1]:
      errors.append(err)
      print(str(iter).zfill(3), ": ", errors[-1])
      t0.append(rotation)
      z = np.matrix(np.zeros((2, 1)))
      yp = transform_data(yp, t0[-1], z)

    scale = [random_scale(x, yp, error_fn) for i in range(n)]
    scale.sort()
    err, scaling = scale[0]
    if err < threshold:
      print(str(iter).zfill(3), ": ", err)
      break
    elif err < errors[-1]:
      errors.append(err)
      print(str(iter).zfill(3), ": ", errors[-1])
      t0.append(scaling)
      z = np.matrix(np.zeros((2, 1)))
      yp = transform_data(yp, t0[-1], z)
  
  return yp 

''' Application starter
'''
def main():
  # generate sample data
  x = generate_data()

  # transform the data: rotate, translate, scale
  t0, t1, k = generate_transformations()
  y = transform_data(x, t0, t1, k=1.0)

  # shuffle points in y matrix
  ys = shuffle_data(y)

  # try to fit point clouds
  yp = fit_point_sets(x, ys, m=30, n=100, error_fn=nearest_squared_error, threshold=1e-3)
  

if __name__ == '__main__':
  main()
