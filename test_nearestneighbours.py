import numpy as np
from sklearn.neighbors import NearestNeighbors

def main():
  X = np.array([[-1, -1, 0, 1], [-2, -1, 0, 2], [-3, -2, 0, 2], [1, 1, 0, 1], [2, 1, 0, 1], [3, 2, 0, 1]])
  Y = np.array([             [-2.5, -1.1, 0, 2], [-3, -2, 0, 1],        [2, 1, 0, 1], [3, 2, 0, 1]])
  print('X:\n', X)
  print('X[:,2]\n', X[:,:2])
  nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(X)
  distances, indices = nbrs.kneighbors(Y)
  print('Indices:\n', indices)
  print('Distances:\n', distances)

if __name__ == '__main__':
  main()