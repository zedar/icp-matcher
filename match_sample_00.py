'''
Iterative Closest Point algorithm finds the best fitting transformation (rotation plus translation) between two point sets.
Uses internally SVD least squared best fit.sum
'''
import rdg.matcher.data_generator as dg
import rdg.matcher.icp as icp
def main():
  # generate sample data
  x = dg.generate_data()
  print('Input matrix X: ', type(x), ' shape:', x.shape)

  # transform the data: rotate, translate, scale
  t0, t1, k = dg.generate_transformations()
  y = dg.transform_data(x, t0, t1, k=1.0)
  print('Input matrix Y: ', type(y), ' shape:', y.shape)

  # shuffle points in y matrix
  ys = dg.shuffle_data(y)

  t, distances, iterations = icp.icp(x, ys, tolerance=1e-9)

  print(t)

if __name__ == '__main__':
  main()