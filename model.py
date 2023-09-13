import numpy as np
from myutils import *
import random
from lossFunction import *

class MatrixFactorization:
    def __init__(self, R, isValidRating, D=10, eta=0.2, w0=1e-6, alpha=1e-6, loadMatrices=False, verbose=True):
        """
        Initialize the MatrixFactorization class.

        Args:
            R (numpy.ndarray): Ratings matrix.
            isValidRating (numpy.ndarray): Matrix indicating valid ratings.
            D (int): Dimensionality of the feature matrices.
            eta (float): Initial learning rate.
            w0 (float): Uknown rating weight parameter of Loss function.
            w0 (float): Regularization parameter of Loss function.
            loadMatrices (bool): Whether to load pre-existing matrices.
            verbose (bool): Whether to print training progress.

        Attributes:
            R (numpy.ndarray): Ratings matrix.
            isValidRating (numpy.ndarray): Matrix indicating valid ratings.
            N (int): Number of users.
            M (int): Number of items.
            D (int): Dimensionality of the feature matrices.
            eta (float): Starting learning rate.
            eta (float): The current learning rate.
            U (numpy.ndarray): User feature matrix.
            V (numpy.ndarray): Item feature matrix.
            LossFunction (LossFunction): Loss function instance.
            Rbar (numpy.ndarray): Predicted ratings matrix.
            validRatings (set): Set of all (i, j) such that R[i][j] is valid
            validRatingsRow (array of sets): validRatingRow[i] is a set of all j such that R[i][j] is valid
            validRatingsCol (array of sets): validRatingRow[j] is a set of all i such that R[i][j] is valid
            verbose (bool): Whether to print training progress.
        """
        self.R = R
        self.D = D
        self.N, self.M = R.shape
        self.isValidRating = isValidRating

        self.initEta = eta
        self.eta = eta

        self.LossFunction = LossFunction(w0=w0, alpha=alpha)
        self.Rbar = None
        self.verbose = verbose

        self.initialize_features(loadMatrices)

        self.validRatings, self.validRatingsRow, self.validRatingsCol = None, None, None
        self.computeValidRatingsSets()


    def initialize_features(self, loadMatrices):
        """
        Initialize feature matrices U and V.

        Args:
            loadMatrices (bool): Whether to load pre-existing matrices.

        Returns:
            tuple: User feature matrix U, Item feature matrix V.
        """
        if loadMatrices:
            U, V = np.load("U.npy"), np.load("V.npy")
            if U.shape != (self.N, self.D) or V.shape != (self.M, self.D):
                raise ValueError("Loaded matrices don't have valid dimensions.")
            self.U, self.V = U, V
        else: 
            U = np.zeros((self.N, self.D))
            for i in range(self.N):
                U[i][random.randint(0, self.D - 1)] = 1

            V = np.zeros((self.M, self.D))
            for j in range(self.M):
                V[j][random.randint(0, self.D - 1)] = 1

            self.U, self.V = U, V

    def train(self, sgd_iterations = 100):
        """ 
        Train the matrix factorization model. 

        Returns:
            (array): An array containing tuples (iteration, Loss) containing the iteration number and corresponding loss
        """

        self.Loss = self.LossFunction.calc_loss(self.U, self.V, self.R, self.validRatings)

        if self.verbose: print(f"L(U, V)/(# of valid ratings) = {self.Loss / len(self.validRatings)}\tL(U, V) = {self.Loss}")

        training_history = []
        iteration = 0
        while iteration < sgd_iterations:
            if self.verbose: print(f"Starting the {iteration + 1}-th learning step. [eta = {self.eta}]")

            jU, jV = self.LossFunction.calc_jacobians(self.U, self.V, self.R, (self.validRatings, self.validRatingsRow, self.validRatingsCol))
            nU = self.U - self.eta * jU
            nV = self.V - self.eta * jV

            nLoss = self.LossFunction.calc_loss(nU, nV, self.R, self.validRatings)

            if nLoss > self.Loss:
                self.eta /= 2
                continue

            if self.verbose: print(f"\tL(U, V)/(# of valid ratings) = {self.Loss / len(self.validRatings)}\tL(U, V) = {self.Loss}")

            self.U, self.V, self.Loss = nU, nV, nLoss

            np.save("U.npy", self.U)
            np.save("V.npy", self.V)
            if self.verbose: print("\tNew U, V computed and saved.")

            training_history.append(self.Loss)
            iteration += 1
        
        self.eta = self.initEta
        self.Rbar = self.U @ self.V.T

        return training_history

    def update_rating(self, I, J, newRating, sgd_iterations = 25):
        """ 
        Update a rating in the dataset and retrain the model. 

        Returns:
            (array): An array where in the i-th position is the Loss corresponding to the i-th iteration
        """
        self.R[I][J] = newRating
        self.isValidRating[I][J] = True
        self.validRatings.add((I, J))
        self.validRatingsRow[I].add(J)
        self.validRatingsRow[J].add(J)


        self.Loss = self.LossFunction.calc_loss(self.U, self.V, self.R, self.validRatings)

        if self.verbose: print(f"L(U, V)/(# of valid ratings) = {self.Loss / len(self.validRatings)}\tL(U, V) = {self.Loss}")

        training_history = []
        iteration = 0
        while iteration < sgd_iterations:
            if self.verbose: print(f"Starting the {iteration + 1}-th update step.")

            jU, jV = self.LossFunction.calc_jacobians(self.U, self.V, self.R, (self.validRatings, self.validRatingRow, self.validRatingsCol))
            nU = self.U - self.eta * jU
            nV = self.V - self.eta * jV

            nLoss = self.L_function(nU, nV, self.R, self.validRatings)

            if nLoss > self.Loss:
                if self.verbose: print("\tStep failed, reducing learning rate.")
                self.eta /= 2
                continue

            if self.verbose: print(f"\tL(U, V)/(# of valid ratings) = {self.Loss / len(self.validRatings)}\tL(U, V) = {self.Loss}")

            self.U, self.V, self.Loss = nU, nV, nLoss

            np.save("U.npy", self.U)
            np.save("V.npy", self.V)
            if self.verbose: print("\tNew U, V computed and saved.")

            training_history.append(self.Loss)
            iteration += 1

        self.eta = self.initEta
        self.Rbar = U @ V.T

    def predict(self, i, j): 
        """ Predict a rating for a user-item pair. """
        return self.Rbar[i][j]


    def computeValidRatingsSets(self):
        """
        Convert the sparse boolean matrix self.isValidRating[*][*] into 
        sets so it is more efficient to do computations with it.

        Updates values to:
            self.validRatings (set):  A set holding the (i, j) tuples where isValidRating[i][j] = True
            self.validRatingsRow (array of sets):  The set self.validRatingsRow[i] holds all j such that isValidRating[i][j] = True
            self.validRatingsCol (array of sets):  The set self.validRatingsCol[j] holds all i such that isValidRating[i][j] = True
        """
        validRatings = set()
        calRrows = [set() for _ in range(len(self.isValidRating))]
        calRcols = [set() for _ in range(len(self.isValidRating[0]))]
        for i in range(len(self.isValidRating)):
            for j in range(len(self.isValidRating[i])):
                if self.isValidRating[i][j]:
                    validRatings.add((i, j)) 
                    calRrows[i].add(j)
                    calRcols[j].add(i)

        self.validRatings, self.validRatingsRow, self.validRatingsCol = (validRatings, calRrows, calRcols)