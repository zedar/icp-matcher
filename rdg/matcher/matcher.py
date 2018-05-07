"""
Match roadagram to HD map features
"""
import numpy as np
from shapely.wkt import loads as loads_wkt
from shapely.geometry import Point, LineString

import rdg.matcher.icp as icp

"""
Match roadagram features to HD map features
Args:
  roadagram: features of roadagram
  hdmap: features of roadagram
Returns:
  Result of matching
"""
def match(roadagram, hdmap):
  S = to_feature_matrix(hdmap)
  M, id_mapping = to_feature_matrix(roadagram, return_id_mapping=True)
  print('Matrix S:\n', S)
  print('Matrix M:\n', M)

  t, distances, iterations, mean_error = icp.icp(M, S, max_iterations=2000, error=1e-2, tolerance=1e-9)

  print('Translation matrix:\n', t)

  aligned_M = np.ones((M.shape[0], M.shape[1]+1))
  aligned_M[:,0:M.shape[1]] = np.copy(M)
  aligned_M = np.dot(t, aligned_M.T)
  aligned_M = aligned_M.T

  R = M.copy()
  R[:,:3] = aligned_M[:,:3]

  print('Aligned M:\n', R)
  aligned_roadagram = roadagram.copy()

  update_features_geometry(aligned_roadagram, R, id_mapping)

  return aligned_roadagram

"""
Convert features (parsed JSON) to numpy matrix of feature vectors.
Args:
  features: array of features
Returns:
  numpy matrix 
"""
def to_feature_matrix(features, return_id_mapping=False):
  pnts = []
  id_mapping = {}
  for f in features['features']:
    # extract feature type
    t = f['feature']
    # extract category
    cat = 'undefined'
    if 'type' in f:
      cat = f['type']
    # extract geometry
    if t == 'RoadSign':
      print(f['geometry'])
      c = list(list(loads_wkt(f['geometry']).coords)[0])
      c.append(10)
      # append category property
      pnts.append(c)
      if return_id_mapping:
        id_mapping[f['id']] = [len(pnts)-1]
    elif t == 'LinearFeature':
      cs = list(loads_wkt(f['geometry']).coords)
      idxs = []
      for c in cs:
        # append category property
        cx = list(c)
        cx.append(20)
        pnts.append(cx)
        idxs.append(len(pnts)-1)
      if return_id_mapping:
        id_mapping[f['id']] = idxs
  if return_id_mapping:
    return np.matrix(pnts), id_mapping
  else:
    return np.matrix(pnts)

"""
Convert matrix of feature vectors to JSON
Args:
  M: matrix of features
Returns:
  JSON with aligned features
"""
def update_features_geometry(features, M, id_mapping):
  fs = features['features']
  for f in fs:
    rows = id_mapping[f['id']]
    m = M[rows, :3]
    if f['feature'] == 'RoadSign':
      print('Matrix with geometry:\n', m)
      f['geometry'] = Point(m[0,0], m[0,1], m[0,2]).wkt
    elif f['feature'] == 'LinearFeature':
      f['geometry'] = LineString(np.asarray(m)).wkt
  
  return features