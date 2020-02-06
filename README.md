# KMeanCosineCluster
This is the first version of KMeanCosineCluster. It's a model to create clusters from a list of vectors using cosine distance instead euclidean

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
cvc = CosineKMean(
    n_iterations = 10, 
    vect_size = 10, 
    nk = 2
)

clusters, kcenters = cvc.find_kcenters(data, col_name='Embeddings')
print(clusters)
print(kcenters)
```
