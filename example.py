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
# non-valid ratings will be exactly the zeroes of R.
# For other default values check the MatrixFactorization object.

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