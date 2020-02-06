class CosineKMean():
  '''
  Class to create a model to clust vectors using cosine distance
  as metric to comput the cluster distance
  '''
  def __init__(self, n_iterations, vect_size, nk=10, perc_stop=0.1):
    '''
    Params:
      - vect_size = umber of dimensions in a vector
      - nk = number of clusters
      -  perc_stop = represent the percentual of changing vectors from one epoch
                     to the other above that keep doing clustering. Below stop
      - n_iterations = how many iteration perform before stopping
    '''
    self.vect_size = vect_size
    self.nk = nk
    self.perc_stop = perc_stop
    self.n_iterations = n_iterations
    self.kcenters = []

  def find_kcenters(
      self, data:pd.DataFrame, col_name='Vectors', similarity=cosine_similarity):
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
    print('\n KCENTERS ARE: \n')
    print(kcenters)

    # Now it's time to create clusters
    for epoch in range(0, self.n_iterations):
      # run an epoch
      changing_rows, data = self.__run_epoch(data, kcenters, similarity, col_name)
      # check stop conditions
      stop = self.__check_stop_conditions(changing_rows, data.shape[0])
      if stop:
        print('Stop clustering cause changed items are less then {}%'.format(
            self.perc_stop * 100))
        break;
      #compute new centroids
      kcenters = self.__replace_centroids(kcenters, data, col_name)
      print('---- NEW CENTROIDS -------')
      print(kcenters)
      # save kcenters 
    return data

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
    for i in range(len(v)):
      v3.append(float((v1[i] + v2[i]))/2)
    return v3

  def __check_stop_conditions(self, changed_data, len_data):
    perc = self.perc_stop
    return changed_data <= (len_data * perc)


  def __run_epoch(self, data: pd.DataFrame, kcenters, similarity, col_name):
    changing_row = 0
    #print('\n ---- EPOCH -----\n')
    for i, row in data.iterrows():
      #print('------------------------\n\n')
      #print(row)
      sim = 0
      old_cluster = row.Cluster
      new_cluster = 0

      for clust in kcenters.keys():
        vect = kcenters[clust]
        #print('Test with k center: ')
        #print(vect, clust)
        tmp_sim = similarity(vect, row[col_name])
        #print('Simialrity: ' + str(tmp_sim))
        new_cluster,sim = (new_cluster,sim) if tmp_sim <= sim else (clust,tmp_sim)
      
      #print('Result: ' + str(new_cluster))
      if new_cluster != old_cluster:
        changing_row += 1
        data.Cluster.iloc[i] = new_cluster
      
    return changing_row, data
