import numpy as np
from myutils import *
import random
from lossFunction import *

class MatrixFactorization:
    def __init__(self, datasetName = None, loadMatrices = False, D=10, eta=0.2, verbose=True):
        """
        Initialize the MatrixFactorization class.

        Args:
            N (int): Number of users.
            M (int): Number of items.
            datasetName (str): Name of the dataset (CSV file).
            loadMatrices (bool): Whether to load pre-existing matrices.
            D (int): Dimensionality of the feature matrices.
            eta (float): Initial learning rate.
            verbose (bool): Whether to print training progress.

        Attributes:
            N (int): Number of users.
            M (int): Number of items.
            D (int): Dimensionality of the feature matrices.
            eta (float): Initial learning rate.
            U (numpy.ndarray): User feature matrix.
            V (numpy.ndarray): Item feature matrix.
            LossFunction (LossFunction): Loss function instance.
            R (numpy.ndarray): Ratings matrix.
            isValidRating (numpy.ndarray): Matrix indicating valid ratings.
            userID (dict): Mapping of user index to user ID.
            itemID (dict): Mapping of item index to item ID.
            getUserFromID (dict): Mapping from user ID to index.
            getItemFromID (dict): Mapping from item ID to index.
            Rbar (numpy.ndarray): Predicted ratings matrix.
            verbose (bool): Whether to print training progress.
        """
        self.N = None
        self.M = None
        self.D = D
        self.eta = eta

        self.LossFunction = LossFunction()

        self.R = None
        self.isValidRating = None
        self.userID, self.itemID = None, None
        self.getUserFromID, self.getItemFromID = None, None
        self.Rbar = None
        self.verbose = verbose

        if datasetName != None:
            self.readDataset(datasetName)

        self.U, self.V = self.initialize_features(loadMatrices)


    def initialize_features(self, loadMatrices):
        """
        Initialize feature matrices U and V.

        Args:
            loadMatrices (bool): Whether to load pre-existing matrices.

        Returns:
            tuple: User feature matrix U, Item feature matrix V.
        """
        if loadMatrices:
            return np.load("U.npy"), np.load("V.npy")

        U = np.zeros((self.N, self.D))
        for i in range(self.N):
            U[i][random.randint(0, self.D - 1)] = 1

        V = np.zeros((self.M, self.D))
        for j in range(self.M):
            V[j][random.randint(0, self.D - 1)] = 1

        return U, V
    
    def readDataset(self, datasetName):
        """
        Read and initialize the dataset.

        Args:
            datasetName (str): Name of the dataset (CSV file).

        Raises:
            ValueError: If the dataset name is not recognized.
        """
        if datasetName[-4:] == ".csv":
            (
                self.R, self.isValidRating, self.userID, self.itemID, 
                self.getUserFromID, self.getItemFromID 
            ) = read_csv(datasetName[:-4])
        else:
            raise ValueError("Dataset file format not supported.")

        self.N = len(self.userID)
        self.M = len(self.itemID)

    def train(self):
        """ Train the matrix factorization model. """
        validRatings, calRrows, calRcols = getValidRatingsSets(self.isValidRating)

        self.Loss = self.LossFunction.calc_loss(self.U, self.V, self.R, validRatings)

        if self.verbose: print(f"L(U, V)/N = {self.Loss / len(validRatings)}")

        _ = 0
        while _ < 100:
            if self.verbose: print(f"Starting the {_ + 1}-th learning step. [eta = {self.eta}]")

            validRatings, calRrows, calRcols = getValidRatingsSets(self.isValidRating)

            jU, jV = self.LossFunction.calc_jacobians(self.U, self.V, self.R, (validRatings, calRrows, calRcols))
            nU = self.U - self.eta * jU
            nV = self.V - self.eta * jV

            nLoss = self.LossFunction.calc_loss(nU, nV, self.R, validRatings)

            if nLoss > self.Loss:
                self.eta /= 2
                continue

            if self.verbose: print(f"\tL(U, V)/N = {nLoss / len(validRatings)}")

            self.U, self.V, self.Loss = nU, nV, nLoss

            np.save("U.npy", self.U)
            np.save("V.npy", self.V)
            if self.verbose: print("\tNew U, V computed and saved.")

            _ += 1

        self.Rbar = self.U @ self.V.T

    def update_rating(self, I, J, newRating):
        """ Update a rating in the dataset and retrain the model. """
        self.R[I][J] = newRating
        self.isValidRating[I][J] = True

        validRatings, calRrows, calRcols = getValidRatingsSets(self.isValidRating)
        validRatings.add((I, J))
        calRrows[I].add(J)
        calRcols[J].add(I)

        _ = 0
        while _ < 25:
            if self.verbose: print(f"Starting the {_ + 1}-th update step.")

            jU, jV = self.LossFunction.calc_jacobians(self.U, self.V, self.R, (validRatings, calRrows, calRcols))
            nU = self.U - self.eta * jU
            nV = self.V - self.eta * jV

            nLoss = self.L_function(nU, nV, self.R, validRatings)

            if nLoss > self.Loss:
                if self.verbose: print("\tStep failed, reducing learning rate.")
                self.eta /= 2
                continue

            np.save("U.npy", self.U)
            np.save("V.npy", self.V)

            if self.verbose: print("\tNew U, V computed and saved.")
            if self.verbose: print(f"\tL(U, V) = {nLoss}")

            self.U, self.V, self.Loss = nU, nV, nLoss

            _ += 1

        self.Rbar = U @ V.T

    def predict(self, i, j): 
        """ Predict a rating for a user-item pair. """
        return self.Rbar[i][j]