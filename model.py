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
            validRatings (set): Set of all (i, j) such that R[i][j] is valid
            validRatingsRow (array of sets): validRatingRow[i] is a set of all j such that R[i][j] is valid
            validRatingsCol (array of sets): validRatingRow[j] is a set of all i such that R[i][j] is valid
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

        self.validRatings = None
        self.validRatingsRow = None
        self.validRatingsCol = None


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

    def train(self, sgd_iterations = 100):
        """ Train the matrix factorization model. """
        self.validRatings, self.validRatingsRow, self.validRatingsCol = getValidRatingsSets(self.isValidRating)

        self.Loss = self.LossFunction.calc_loss(self.U, self.V, self.R, self.validRatings)

        if self.verbose: print(f"L(U, V)/N = {self.Loss / len(self.validRatings)}")

        _ = 0
        while _ < sgd_iterations:
            if self.verbose: print(f"Starting the {_ + 1}-th learning step. [eta = {self.eta}]")

            jU, jV = self.LossFunction.calc_jacobians(self.U, self.V, self.R, (self.validRatings, self.validRatingsRow, self.validRatingsCol))
            nU = self.U - self.eta * jU
            nV = self.V - self.eta * jV

            nLoss = self.LossFunction.calc_loss(nU, nV, self.R, self.validRatings)

            if nLoss > self.Loss:
                self.eta /= 2
                continue

            if self.verbose: print(f"\tL(U, V)/N = {nLoss / len(self.validRatings)}")

            self.U, self.V, self.Loss = nU, nV, nLoss

            np.save("U.npy", self.U)
            np.save("V.npy", self.V)
            if self.verbose: print("\tNew U, V computed and saved.")

            _ += 1

        self.Rbar = self.U @ self.V.T

    def update_rating(self, I, J, newRating, sgd_iterations = 25):
        """ Update a rating in the dataset and retrain the model. """
        self.R[I][J] = newRating
        self.isValidRating[I][J] = True
        self.validRatings, self.validRatingsRow, self.validRatingsCol = getValidRatingsSets(self.isValidRating)

        self.Loss = self.LossFunction.calc_loss(self.U, self.V, self.R, self.validRatings)

        if self.verbose: print(f"L(U, V)/N = {self.Loss / len(self.validRatings)}")

        _ = 0
        while _ < sgd_iterations:
            if self.verbose: print(f"Starting the {_ + 1}-th update step.")

            jU, jV = self.LossFunction.calc_jacobians(self.U, self.V, self.R, (self.validRatings, self.validRatingRow, self.validRatingsCol))
            nU = self.U - self.eta * jU
            nV = self.V - self.eta * jV

            nLoss = self.L_function(nU, nV, self.R, self.validRatings)

            if nLoss > self.Loss:
                if self.verbose: print("\tStep failed, reducing learning rate.")
                self.eta /= 2
                continue

            if self.verbose: print(f"\tL(U, V)/N = {nLoss / len(self.validRatings)}")

            self.U, self.V, self.Loss = nU, nV, nLoss

            np.save("U.npy", self.U)
            np.save("V.npy", self.V)
            if self.verbose: print("\tNew U, V computed and saved.")

            _ += 1

        self.Rbar = U @ V.T

    def predict(self, i, j): 
        """ Predict a rating for a user-item pair. """
        return self.Rbar[i][j]


    def getValidRatingsSets(isValidRating):
        """
        Convert the sparse boolean matrix isValidRating[*][*] into 
        sets so it is more efficient to do computations with it.

        Returns:
            validRatings (set):  A set holding the (i, j) tuples where isValid[i][j] = True
            calRrows (array of sets):  The set calRrows[i] holds all j such that isValid[i][j] = True
            calRcols (array of sets):  The set calRrows[j] holds all i such that isValid[i][j] = True
        """
        validRatings = set()
        calRrows = [set() for _ in range(len(isValidRating))]
        calRcols = [set() for _ in range(len(isValidRating[0]))]
        for i in range(len(isValidRating)):
            for j in range(len(isValidRating[i])):
                if isValidRating[i][j]:
                    validRatings.add((i, j)) 
                    calRrows[i].add(j)
                    calRcols[j].add(i)
        return (validRatings, calRrows, calRcols)