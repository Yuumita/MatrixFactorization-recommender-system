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

The loss function we seek to minimize is

$$ L(U, V) = \sum_{(i,j)\in \mathcal{R}}\left( r_{ij} - \mathbf{u}_ {i} \mathbf{v}_ {j}^\text{T} \right)^{2} + w_{0}\sum_{(i,j) \not\in \mathcal{R}}+  \left( \mathbf{u}_ {i}\mathbf{v}_ {j}^\text{T}  \right)^{2},  \quad w_{0}\in\mathbb{R}\ $$

where $\\{\mathbf{u}_ i\\}_ {i=1}^N,\  \\{\mathbf{v}_ j\\}_ {j=1}^M$ are the features of the users and movies respectively.

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
import numpy as np
from model import MatrixFactorization

# A "random" rating matrix with 3 users and 5 items
R = np.array([
    [3, 1, 4, 0, 5],
    [9, 0, 6, 5, 3],
    [0, 8, 9, 7, 0]
])

# A boolean matrix describing the validRating
isValidRating = np.array([
    [True, True, True, False, True],
    [True, False, True, True, True],
    [False, True, True, True, False]
])


mf_model = MatrixFactorization(
  R, isValidRating=isValidRating, 
  D=2, eta=0.2, w0=1e-6, alpha=1e-6, 
  loadMatrices=False, verbose=True
)
# If the isValidRating matrix is not given, the 
# non-valid ratings will be exactly the zeroes of R

mf_model.train(sgd_iterations=20)
# Since we turned on the mf_model.verbose
# it will print all the details of the training

print(mf_model.U)
print(mf_model.V)
print(mf_model.Rbar)

# The initial U, V features are random, but a possible print could be the following:
'''
[[ 1.34895964  1.101414  ]
 [ 2.83583722  1.28456064]
 [-0.43182457  3.84576319]]

[[ 2.67867176  0.46546588]
 [-0.45758775  1.90956145]
 [ 0.91859405  2.30614113]
 [ 0.81116807  1.8178006 ]
 [ 1.09155218  0.90104751]]

[[4.12609071 1.48595031 3.77916243 3.09638402 2.46488617]
 [8.19419623 1.15530312 5.56736134 4.63541571 4.25291446]
 [0.63335525 7.54131876 8.47220118 6.64054835 2.9938563 ]]
 '''
```
