import numpy as np

class LossFunction:

    def __init__(self, w0=1e-6, alpha=1e-6):
        """
        Initialize the LossFunction with parameters.

        Args:
            w0 (float): Uknown rating weight parameter.
            alpha (float): Regularization parameter.
        """
        self.w0 = w0
        self.alpha = alpha

    def calc_jacobians(self, U, V, r, calR):
        """
        Calculate the Jacobians of the loss function with respect to U and V.

        Args:
            U (numpy.ndarray): User matrix.
            V (numpy.ndarray): Item matrix.
            r (numpy.ndarray): Ratings matrix.
            calR (tuple): Tuple containing validRatings, calRrows, and calRcols.

        Returns:
            tuple: Jacobians with respect to U and V (jU, jV).
        """
        N, D = U.shape
        M, Dtmp = V.shape

        if D != Dtmp: 
            raise ValueError("Dimensions of U, V don't match")

        # Jacobians
        jU, jV = np.zeros((N, D)), np.zeros((M, D))

        Sv = np.zeros((D, D))
        Su = np.zeros((D, D))

        for j in range(M):
            Sv += V[[j]].T @ V[[j]]
        for i in range(N):
            Su += U[[i]].T @ U[[i]]

        validRatings, calRrows, calRcols = calR

        # jU
        for k in range(N):
            for j in calRrows[k]:
                jU[k] += -2 * r[k][j] * V[j]
                jU[k] += 2 * (U[k] @ V[j].T) * V[j]
                jU[k] -= (2 * self.w0) * (U[k] @ V[j].T) * V[j]
            jU[[k]] += (2 * self.w0) * (U[[k]] @ Sv)

            # Regularization
            jU[[k]] += self.alpha * 2 * U[[k]]

        # jV
        for k in range(M):
            for i in calRcols[k]:
                jV[k] += -2 * r[i][k] * U[i]
                jV[k] += 2 * (V[k] @ U[i].T) * U[i]
                jV[k] -= (2 * self.w0) * (V[k] @ U[i].T) * U[i]
            jV[[k]] += (2 * self.w0) * (V[[k]] @ Su)

            # Regularization
            jV[[k]] += self.alpha * 2 * V[[k]]

        return (jU, jV)

    def calc_jacobians_at(self, U, V, r, calR, I, J):
        """
        Calculate the gradients of the loss function with respect to user I and item J.

        Args:
            U (numpy.ndarray): User matrix.
            V (numpy.ndarray): Item matrix.
            r (numpy.ndarray): Ratings matrix.
            calR (tuple): Tuple containing validRatings, calRrows, and calRcols.
            I (int): Index for user.
            J (int): Index for item.

        Returns:
            tuple: The gradients of the function with respect to I and J respectively
        """
        N, D = U.shape
        M, Dtmp = V.shape

        if D != Dtmp: 
            raise ValueError("Dimensions of U, V don't match")

        # vector-Jacobians (gradients)
        jUI, jVJ = np.zeros((1, D)), np.zeros((1, D))

        validRatings, calRrows, calRcols = calR

        # jUI
        tmpS = np.zeros((D, D))
        for j in calRrows[I]:
            jUI += -2 * r[I][j] * V[j]
            jUI += 2 * (U[I] @ V[j].T) * V[j]

        # jVJ
        tmpS = np.zeros((D, D))
        for i in calRcols[J]:
            jVJ += -2 * r[i][J] * U[i]
            jVJ += 2 * (V[J] @ U[i].T) * U[i]

        return (jUI, jVJ)

    def calc_loss(self, U, V, r, validRatings):
        """
        Calculate the loss function.

        Args:
            U (numpy.ndarray): User matrix.
            V (numpy.ndarray): Item matrix.
            r (numpy.ndarray): Ratings matrix.
            validRatings (list): List of valid rating pairs.

        Returns:
            float: Loss value.
        """
        N, M, D = len(U), len(V), len(U[0])

        Sv = np.zeros((D, D))
        Su = np.zeros((D, D))

        for j in range(M):
            Sv += V[[j]].T @ V[[j]]
        for i in range(N):
            Su += U[[i]].T @ U[[i]]

        ret = 0
        for p in validRatings:
            i, j = p
            ret += (r[i][j] - U[i] @ V[j].T) ** 2
            ret -= self.w0 * (U[i] @ V[j].T) ** 2

        for i in range(N):
            ret += self.w0 * (U[[i]] @ Sv @ U[[i]].T)
            ret += U[i] @ U[i].T
        for j in range(M):
            ret += V[j] @ V[j].T

        return ret
