import numpy as np
from myutils import *
import random
from model import *

def testModel(mf):
    """
    Test the model by evaluating its predictions.

    Args:
        mf (MatrixFactorization): An instance of the MatrixFactorization class.

    This function evaluates the model's performance by comparing its predictions
    (stored in mf.Rbar) with the actual ratings in the test dataset. It calculates
    three Mean Squared Error (MSE) values for different types of predictions:
    - MSE for all predictions rated as 7/10
    - MSE for all predictions rated as 5/10
    - MSE for predictions made by the model

    The function prints the calculated MSE values for evaluation.

    Returns:
        tuple: The three MSE values as described above.
    """
    mf.Rbar = mf.U @ mf.V.T

    R, validRating, userID, itemID, _, _ = read_csv("test_dataset")

    Loss = 0
    randLoss = 0
    meanLoss = 0

    totC = 0
    for i in range(len(userID)):
        for j in range(len(itemID)):
            if not validRating[i][j]: continue
            if userID[i] in mf.getUserFromID and itemID[j] in mf.getItemFromID:
                utId = mf.getUserFromID[userID[i]]
                itId = mf.getItemFromID[itemID[j]]

                if not mf.isValidRating[utId][itId]:
                    error = abs(R[i][j] - mf.Rbar[utId][itId])

                    Loss += 1 * (R[i][j] - max(min(mf.Rbar[utId][itId], 5), 0))**2
                    randLoss += 1 * (R[i][j] - 2.5)**2
                    meanLoss += 1 * (R[i][j] - 3.5)**2
                    totC += 1

    print("All 7/10 predicitons MSE: \t", meanLoss/totC)
    print("All 5/10 predictions MSE: \t", randLoss/totC)
    print("MF Model predictions MSE: \t", Loss/totC)

    return (meanLoss/totC, randLoss/totC, Loss/totC)




loadMatrices = True if int(input("Load U, V (1) or intialize new ones (2)? ")) == 1 else False

mf_model = MatrixFactorization("train_dataset.csv", D=10, loadMatrices = loadMatrices)

print(f"U.shape = {mf_model.U.shape}, V.shape = {mf_model.V.shape}")

while True: 
    option = int(input("Choose an option (1/2/3)\n\t1. Train model\n\t2. Update a Rating\n\t3. Test Model\n\t(Anything else to quit)\n"))
    if option == 1:
        mf_model.train()
    elif option == 2:
        print("Insert user id, movie id, new rating:\n")
        i = mf_model.getUserFromID[int(input())]
        j = mf_model.getItemFromID[int(input())]
        r = int(input())
        mf_model.update_rating(i, j, r)
    elif option == 3:
        testModel(mf_model)
    else: 
        break
