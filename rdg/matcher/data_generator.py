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
  #t1 = np.matrix(scipy.stats.norm(loc=3, scale=3).rvs(size=(3,2)))
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

''' 
Generate sample data using sinus function in range
Args:
  start: first x-axis value
  end: last x-axis value
  step: xs generated by step starting from `start`
Returns:
  NumPy matrix with rows=number of points and columns(3)=point coordinates (x y z)
'''
def generate_data_sin(start=0.0, end=3.0, step=0.2):
  t = np.arange(start, end, step)
  ty = np.sin(t)
  
  a = np.array([t, ty])
  # 2 additional columns. 1st=z-axis, 2nd=category
  b = np.ones((a.shape[1], a.shape[0]+2))
  b[:,:-2] = np.copy(a.T)
  b[:,-1] = 10  # category 10
  tsx = 1.5
  tsy = np.sin(1.5)-0.2
  b = np.vstack([b, [tsx, tsy, 1.0, 20]]) # z=1.0, category=20
  b = np.matrix(b)
  return b
  #return np.matrix([t, ty]).T

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

""" 
Transform sample data by rotation, translation and scaling
Args:
  x: matrix of data to transform
  t0: rotation matrix
  t1: translation matrix
  k: scaling factor
Returns:
  matrix of transformed data
"""
def transform_data(x, t0, t1, k=1.0):
  # extract first 2 features (x,y) coordinates and apply transformation to
  s = np.zeros((x.shape[0], 2))
  s[:,:] = np.copy(x[:,:2])
  

  # apply an affine transformation to x
  #y = x * t0.T
  y = s * t0.T
  #print('x shape: ', x.shape)
  #print('y shape: ', y.shape)
  #print('t1 shape: ', t1.shape)
  y[:,0] += t1[0,0]
  y[:,1] += t1[1,0]
  y = y*k

  z = x.copy()
  z[:,:2] = np.copy(y)
  return z
  #return y