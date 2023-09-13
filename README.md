# MatrixFactorization-recommender-system
A python implementation of a recommender system with Matrix factorization (Collaborative Filtering). 


## Description of the algorithm

Here we use a user-item relationship with the $U$ - $V$ feature matrices with dimensionality $D$ (which is a parameter of the model). In other words, $U\in\mathbb{R}^{N\times D}$ and $V\in\mathbb{R}^{M\times D}$). 

The algorithm is designed to be efficient for sparse matrices for the ratings $R$. In other words, the set $\mathcal{R}$ of all valid ratings (i.e ratings that the users have given) has a size a lot less than $N\cdot M$ (where $N$ is the user count and $M$ is the item count). 

Namely, we have the following complexities
- **Space complexity:** $O\left( NM + D(N+M) \right)$
- **Jacobian calculation time complexity:** $O\left(|\mathcal{R}|D + (N+M)D^2\right)$
- **Loss calculation time complexity:** $O\left(|\mathcal{R}|D + (N+M)D^2\right)$
- **Precalculations time complexity**: $O(NM)$



## Implementation details

### File description

The main files are 

- **model.py**, containing the MatrixFactorization object
- **lossFunction.py**, containing the LossFunction object with the loss and jacobian of loss calculation functions

The following files are optional

- **myutils.py**, containing some function to read `.csv` files
- **train.py**, which is a file to train and test a MF model with the datasets
  - **train_dataset.csv** which contains some ratings (around 75% of the total ratings)
  - **test_dataset.csv** which has all the ratings (including all of those in **train_dataset.csv**, even though **train.py** would still work if that wasn't the case)

### Example Usage


``` python
(TODO)
```
