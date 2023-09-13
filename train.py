import numpy as np
from myutils import *
import random
from model import *

def testModel(mf, info, test_info):
    """
    Test the model by evaluating its predictions.

    Args:
        mf (MatrixFactorization): An instance of the MatrixFactorization model.
        info (UserItemInfo): An instance of UserItemclass holding the user/item information id's of the train_dataset.
        test_info (UserItemInfo): An instance of UserItemclass holding the user/item information id's of the test_dataset.

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

    # R, validRating, userID, itemID, _, _ = read_csv("test_dataset")

    loss = 0
    randLoss = 0
    meanLoss = 0

    count = 0
    for i in range(len(test_info.userID)):
        for j in range(len(test_info.itemID)):
            if not test_info.isValidRating[i][j]: continue
            if test_info.userID[i] in info.getUserFromID and test_info.itemID[j] in info.getItemFromID:
                utId = info.getUserFromID[test_info.userID[i]]
                itId = info.getItemFromID[test_info.itemID[j]]

                if not info.isValidRating[utId][itId]:
                    error = abs(test_info.R[i][j] - mf.Rbar[utId][itId])

                    loss += 1 * (test_info.R[i][j] - max(min(mf.Rbar[utId][itId], 5), 0))**2
                    randLoss += 1 * (test_info.R[i][j] - 2.5)**2
                    meanLoss += 1 * (test_info.R[i][j] - 3.5)**2
                    count += 1

    print("All 7/10 predicitons MSE: \t", meanLoss/count)
    print("All 5/10 predictions MSE: \t", randLoss/count)
    print("MF Model predictions MSE: \t", loss/count)

    return (meanLoss/count, randLoss/count, loss/count)


test_info = UserItemInfo("test_dataset.csv")
info = UserItemInfo("train_dataset.csv")

loadMatrices = True if int(input("Load U, V (1) or intialize new ones (2)? ")) == 1 else False

mf_model = MatrixFactorization(info.R, info.isValidRating, D=8, loadMatrices = loadMatrices)

print(f"U.shape = {mf_model.U.shape}, V.shape = {mf_model.V.shape}")

while True: 
    option = int(input("Choose an option (1/2/3)\n\t1. Train model\n\t2. Update a Rating\n\t3. Test Model\n\t(Anything else to quit)\n"))
    if option == 1:
        mf_model.train()
    elif option == 2:
        print("Insert user id, movie id, new rating:\n")
        i = info.getUserFromID[int(input())]
        j = info.getItemFromID[int(input())]
        r = float(input())
        mf_model.update_rating(i, j, r)
    elif option == 3:
        testModel(mf_model, info, test_info)
    else: 
        break
