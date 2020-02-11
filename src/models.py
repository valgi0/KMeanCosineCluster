import numpy as np
import pandas as pd
import pickle as p
import zipfile as zf
from scipy import spatial as sp
from sklearn import preprocessing
from tqdm.autonotebook import tqdm, trange
import random as r



class CosineKMeanOptimizedV2():
  '''
  Class to create a model to clust vectors using cosine distance
  as metric to comput the cluster distance
  '''
  def __init__(self, vect_size, nk=10, matrix_size=None):
    '''
    Params:
      - vect_size = umber of dimensions in a vector
      - nk = number of clusters
      
      - matrix_size = how big must be the matrix to put data on cuda memory. Matrix_size = col * row
    '''
    self.vect_size = vect_size
    self.nk = nk
    self.kcenters = dict()
    self.matrix_size = matrix_size
    self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

  def set_kcenters(self, kcenters):
    if type(kcenters) not in [dict]:
      raise Exception('Type not supported, Supported types are: ' + str([dict]))
    nk = len(kcenters)
    if nk != self.nk:
      raise Exception('The number of vectors must be {} but found {}: '.format(self.nk, nk))
    internal_dict = dict()
    for k in kcenters.keys():
      vect = kcenters[k]
      dim = len(vect)
      if dim != self.vect_size:
        raise Exception('The vector size must be {} but found {} at key {}'.format(self.vect_size, dim, k))
      internal_dict[k] = kcenters[k].tolist()
    self.kcenters = internal_dict

  def classify_vects(self, vectors):
    if type(vectors) not in [torch.Tensor, np.array]:
      raise Exception('Type not supported, Supported types are: ' + str([torch.Tensor, np.array]))
    if vectors.ndim != 2:
      raise Exception('Data must have two dimensions')
    nv, dim = vectors.shape
    if dim != self.vect_size:
      raise Exception('The vector size must be {} but found {} '.format(self.vect_size, dim))
    return self.__classify_vects(vectors)
    

  def find_kcenters(
    self, data:pd.DataFrame,
    col_name='Vectors',
    n_iterations = 10, 
    vect_size = 30, 
    perc_stop=0.1,
    nk = 100):
    '''
    Method to create clusters centers using data and Kmean algorithm to aglomerate them.
    Each items in the data is placed in a cluster according its cosine distance to the center
    of the cluster.
    At the beginning the clusters are selected from data randomly but after each
    iteration they are improved computing the sum of the vectors in the in the cluster
    parameters:
      - data : pandas dataframe containing data but a columns called 'Vectors' or col_name
               that contains vectors
      - col_name: nome of column in dataframe that contains vectors
      - similarity: in this version default option is the only supported
      -  perc_stop = represent the percentual of changing vectors from one epoch
                     to the other above that keep doing clustering. Below stop
      - n_iterations = how many iteration perform before stopping
    '''
    row, col = data.shape
    cluster_col = np.zeros(row, dtype=np.int32)
    data['Cluster'] = cluster_col
    kc = set()
    kcenters = dict()

    # computing random centroids
    while(len(kc) != self.nk):
      kc.add(r.randint(0, row - 1))
    kc = [data.iloc[k] for k in list(kc)]
    for i in range(0, len(kc)):
      kcenters[i+1] = kc[i][col_name]

    # Now it's time to create clusters
    for epoch in trange(0, n_iterations, desc='Clustering processing'):

      changing_rows, data = self.__run_epoch(data, kcenters,  col_name)
      # check stop conditions
      stop = self.__check_stop_conditions(changing_rows, data.shape[0], perc_stop)
      if stop:
        print('Stop clustering cause changed items are less then {}%'.format(
            self.perc_stop * 100))
        break;

      #compute new centroids
      kcenters = self.__replace_centroids(kcenters, data, col_name)
      
      # save kcenters 
      self.kcenters = kcenters
    return data, kcenters
  
  def __replace_centroids(self, kcenters, data, col_name):
    for _, row in data.iterrows():
      vect = row[col_name]
      clust = row.Cluster
      kcenters[clust] = self.__sum_vect_and_norm(kcenters[clust], vect)
    return kcenters
  
  def __sum_vect_and_norm(self, v1, v2):
    if len(v1) != len(v2):
      raise Exception('Vectors with different size can\'t be summed')
    v3 = []
    for i in range(len(v1)):
      v3.append(float((v1[i] + v2[i]))/2)
    return v3

  def __check_stop_conditions(self, changed_data, len_data, perc_stop):
    perc = perc_stop
    return changed_data <= (len_data * perc)
  
  def __classify_vects(self, vectors):
    print(self.kcenters)
    m_data = torch.tensor(vectors, dtype=torch.float32).to(device = self.device)
    m_k = torch.tensor(list(self.kcenters.values()), dtype=torch.float32).to(device=self.device)
    #multiplications
    m_mul = torch.matmul(m_data, m_k.t())
    # divisors
    div_data = (m_data * m_data).sum(dim=1)
    div_k = (m_k * m_k).sum(dim=1)
    div_data = torch.sqrt(div_data)
    div_k = torch.sqrt(div_k)
    step1 = m_mul.t() / div_data
    sim = (step1.t() / div_k)
    _ , max_ind= torch.max(sim, 1)
    clusters = []
    for i,row in enumerate(sim):
      index = max_ind[i]
      cluster = list(kcenters.keys())[index]
      clusters.append(cluster)
    return clusters

  def __run_epoch(self, data: pd.DataFrame, kcenters, col_name):
    changing_row = 0
    if self.matrix_size != None:
      if len(data[col_name]) * len(data[col_name].values[0]) > self.matrix_size:
        raise Exception('Data bigger than matrix are not actually supported')

    m_data = torch.tensor(data[col_name].values.tolist(), dtype=torch.float32).to(device = self.device)
    m_k = torch.tensor(list(kcenters.values()), dtype=torch.float32).to(device=self.device)
    #multiplications
    m_mul = torch.matmul(m_data, m_k.t())
   
    # divisors
    div_data = (m_data * m_data).sum(dim=1)
    div_k = (m_k * m_k).sum(dim=1)
    div_data = torch.sqrt(div_data)
    div_k = torch.sqrt(div_k)
    step1 = m_mul.t() / div_data
    sim = (step1.t() / div_k)
    _ , max_ind= torch.max(sim, 1)
    for i,row in enumerate(sim):
      index = max_ind[i]
      cluster = list(kcenters.keys())[index]
      if data.Cluster.iloc[i] != cluster:
        data.Cluster.iloc[i] = cluster
        changing_row += 1
    return changing_row, data                      
