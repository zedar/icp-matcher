import numpy as np
import scipy.stats
import random

''' Generate random rotation and translation matrix as well as scaling factor
:return: tuple of rotation matrix, translation matrix, scaling factor
'''
def generate_transformations():
  # rotate sample data by angle theta, thats value is between 0 and 2PI
  #theta = scipy.stats.uniform(0.0, 2.0*np.pi).rvs()
  theta = scipy.stats.uniform(0.0, 0.6).rvs()
  print('Rotation angle rad:', theta, ' degs: ', np.rad2deg(theta))

  # calculate rotation matrix
  c, s = np.cos(theta), np.sin(theta)
  t0 = np.matrix([[c,-s], [s,c]])
  print('Rotation matrix: \n', t0)

  # calculate translation vector
  t1 = np.matrix(scipy.stats.norm(loc=3, scale=3).rvs(size=(3,2)))
  t1 = np.matrix(scipy.stats.norm(loc=0.2, scale=0.1).rvs(size=(3,2)))
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

''' Generate sample sin data
'''
def generate_data_sin():
  t = np.arange(0, 3, 0.2)
  ty = np.sin(t)
  a = np.array([t, ty])
  b = np.ones((a.shape[1], a.shape[0]+1))
  b[:,:-1] = np.copy(a.T)
  tsx = 1.5
  tsy = np.sin(1.5)-0.2
  b = np.vstack([b, [tsx, tsy, 2.0]])
  print('B.shape:', b.shape)
  return np.matrix([t, ty]).T

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