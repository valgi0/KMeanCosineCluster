# KMeanCosineCluster
This is the first version of KMeanCosineCluster. It's a model to create clusters from a list of vectors using cosine distance instead euclidean
It's check if GPU is available and in the case it is it use the GPU for speed the process up.
Futhermore it provides a fast method to classify vectors according to model internal centroids.
Interal centroids are compute in two way:
* implicitily: during run epochs
* explicitily: using method *set_kcenters*

## Data
Data must be put in a Pandas Dataframe and all vectors must be in a column.

## Example

Create data
```
d = np.random.randint(-10,10, 100)
d = d.reshape((10,10))
data = pd.DataFrame(columns=['Embeddings'])
for i,v in enumerate(d):
  data.loc[i] = [v]
data
```

Run the model

```
cvc = CosineKMeanOptimizedV2(
    vect_size = 10, 
    nk = 2
)

clusters, kcenters = cvc.find_kcenters(data, col_name='Embeddings', n_iteration = 10)
print(clusters)
print(kcenters)
```

Create data
```
kc_tmp = torch.randint(-1,2,(3,3))
data = torch.randint(-2,4, (10,3))
kcenters = dict()
for i,v in enumerate(kc_tmp):
  kcenters[i+1] = v
print(kcenters)
print(data)
```
Classify them
```
model = CosineKMeanOptimizedV2( 
    vect_size = 3, 
    nk = 3
)
model.set_kcenters(kcenters)
model.kcenters

model.classify_vects(data)
```
