import csv
import numpy as np
import random

def coordinate_compress(A):
    """
    Returns the compression of array A with the function that maps the new values to the old ones

    Parameters:
    A (array): The array to compress

    Returns:
    (array, dict()): The array and a function that maps the new values to the old ones
    """

    B = sorted(A)
    f, finv, C = dict(), dict(), 0
    for i in range(len(B)):
        if B[i] not in f:
            f[B[i]] = C
            finv[C] = B[i]
            C += 1
    return ([f[e] for e in A], finv, f) 

def read_csv(name):
    """ 
    Return an array with the ratings, a truth array having the valid ratings 
    and the inverse map of coordinate compression.

    Returns:
        R (array): The ratings read from the csv file
        T (array): T[i][j] = True iff R[i][j] is valid
        userID (dict()): userID[i] is the ID of the i-th user
        itemID (dict()): itemID[j] is the ID of the j-th item
        user (dict()): user[i] is the user with id i
        item (dict()): item[j] is the item with id j
    """
    inUsers = []
    inRatings = []
    inItems = []
    inTimestamps = []


    # opening the CSV file
    with open(f'{name}.csv', mode ='r') as file:
    
        # reading the CSV file
        csvFile = list(csv.reader(file))
        
        # displaying the contents of the CSV file
        for lines in csvFile[1:]:
            inUsers.append(int(lines[0]))
            inItems.append(int(lines[1]))
            inRatings.append(float(lines[2]))
            inTimestamps.append(int(lines[3]))

        inUsers, userID, user = coordinate_compress(inUsers)
        inItems, itemID, item = coordinate_compress(inItems)

        N, M = len(userID), len(itemID)
        R = np.zeros((N, M))
        valid = np.full((N, M), False, dtype=bool)

        for i in range(len(csvFile[1:])):
            R[inUsers[i]][inItems[i]] = inRatings[i]
            valid[inUsers[i]][inItems[i]] = True
    
        return (R, valid, userID, itemID, user, item)
