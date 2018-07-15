#!/usr/bin/env python

import numpy as np
import Point

def printMatrix(matrix):
    (row, col) = matrix.shape
    for i in range(row):
        for j in range(col):
            print matrix[i, j], "  ", 
        print ""
    print ""

def invertible(matrix):
    (row, col) = matrix.shape
    if (row != col):
        return False
    inverse = np.linalg.inv(matrix)
    identity = np.diag(np.ones(row))
    eps = 1.0e-8
    return np.linalg.norm(inverse.dot(matrix) - identity) < eps

def readPoints(inputFileName):
    import os
    assert(os.path.exists(inputFileName))
    points = []
    ifile = open(inputFileName, "r")
    for (index, string) in enumerate(ifile):
        a = string.split()
        x = float(a[0])
        y = float(a[1])
        label = int(a[2])
        points.append(Point.Point(x, y, label))
    ifile.close()
    return points

def generateMatrix(points):
    dimension = len(points)
    A = np.zeros((dimension, dimension))
    for i in range(dimension):
        for j in range(i, dimension):
            A[i, j] = points[i].label * points[j].label * (Point.inner(points[i], points[j]))
            if (i != j):
                A[j, i] = A[i, j]
    return A

def gradient(A, alpha, t, w, y, C):
    assert(C > 0)
    assert(t > 1.0)
    N = len(alpha)
    assert(len(y) == N)
    (row, col) = A.shape
    assert(N == row)
    assert(row == col)
    result = np.zeros(N+1)
    vector = A.dot(alpha)
    for i in range(N):
        result[i] = -1.0 + vector[i] - 1.0/t*(1./alpha[i] - 1./(C - alpha[i])) + w*y[i]
    result[-1] = alpha.dot(y)
    return result

def Hessian(A, alpha, t, y, C):
    N = len(alpha)
    (row, col) = A.shape
    assert(N == row)
    assert(row == col)
    assert(t > 1.0)
    assert(C > 0)
    result = A
    vector = np.zeros(N)
    for i in range(N):
        vector[i] = 1.0/t*(1.0/alpha[i]**2 + 1.0/(C - alpha[i])**2)
    result = result + np.diag(vector)
    H = np.zeros((N+1, N+1))
    H[0:-1, 0:-1] = result
    H[-1, -1] = 0
    H[-1, 0:-1] = y
    H[0:-1, -1] = y
    return H

def combine(alpha, w):
    N = len(alpha)
    xi = np.zeros(N+1)
    xi[0:-1] = alpha
    xi[-1] = w
    return xi

def Newton(alpha0, w0, A, t, y, C, iprint = False):
    xi0 = combine(alpha0, w0)
    xi = xi0
    iterationMax = 20
    eps = 1.0e-10
    count = 0
    while(count <= iterationMax):
        count = count + 1
        H = Hessian(A, xi0[0:-1], t, y, C)
        #printMatrix(H)
        if (not invertible(H)):
            print "Warning: Hessian matrix is singular. "
        xi = xi0 - np.linalg.inv(H).dot(gradient(A, xi0[0:-1], t, xi0[-1], y, C))
        error = np.linalg.norm(xi - xi0)
        if (error < eps):
            break
        if (iprint):
            print "count = ", count, ", error = ", error
        xi0 = xi
    return xi0

def Solver(alpha0, w0, t0, A, y, C, iprint = False):
    assert(t0 >= 1.0)
    upperBound = 2.0e5*t0
    factor = 1.08
    solution0 = combine(alpha0, w0)
    eps = 9.0e-6
    t = []
    t.append(t0)
    while(t0 < upperBound):
        t0 = factor*t0
        t.append(t0)
        solution = Newton(solution0[0:-1], solution0[-1], A, t0, y, C, iprint)
        error = np.linalg.norm(solution - solution0)
        print "t = ", t0, ", error = ", error
        if (error < eps):
            break
        solution0 = solution
    return solution0

def quicksort(t):
    if (len(t) == 0 or len(t) == 1):
        return t
    pivot = t[0][0]
    lower = []
    equal = []
    upper = []
    for i in range(len(t)):
        if (t[i][0] < pivot):
            lower.append(t[i])
        elif(t[i][0] == pivot):
            equal.append(t[i])
        else:
            upper.append(t[i])
    lower = quicksort(lower)
    equal = quicksort(equal)
    upper = quicksort(upper)
    result = []
    for i in range(len(lower)):
        result.append(lower[i])
    for i in range(len(equal)):
        result.append(equal[i])
    for i in range(len(upper)):
        result.append(upper[i])
    return result

def svm(negativeFileName, positiveFileName, C):
    import os
    import random

    assert(os.path.exists(negativeFileName))
    assert(os.path.exists(positiveFileName))
    negativePoints = readPoints(negativeFileName)
    positivePoints = readPoints(positiveFileName)

    points = []
    for i in range(len(negativePoints)):
        points.append(negativePoints[i])
    for i in range(len(positivePoints)):
        points.append(positivePoints[i])

    point_x = []
    for i in range(len(points)):
        point_x.append(points[i].x)
    xLower = min(point_x)
    xUpper = max(point_x)

    A = generateMatrix(points)
    if (False):
        print "A = "
        printMatrix(A)
    t0 = 1.0
    y = np.zeros(len(points))
    for i in range(len(y)):
        y[i] = points[i].label
    alpha0 = np.ones(len(points))
    for i in range(len(alpha0)):
        alpha0[i] = 0.05 #random.uniform(0.99, 1.5)
    w0 = 1.0 #random.uniform(-1, 1)
    solution = Solver(alpha0, w0, t0, A, y, C, False)
    alpha = solution[0:-1]
    w = solution[-1]
    print "alpha = ", alpha
    print "w = ", w
    beta = np.zeros(2)
    for i in range(len(points)):
        vector = np.zeros(2)
        vector[0] = points[i].x
        vector[1] = points[i].y
        beta = beta + solution[i]*y[i]*vector
    print "beta = ", beta
    print "alpha*y = ", alpha.dot(y)
    t = []
    for i in range(len(alpha)):
        t.append((alpha[i], points[i]))
    t = quicksort(t)
    t.reverse()
    ofile = open("result.txt", "w")
    for i in range(len(t)):
        ofile.write(str(t[i][0]) + "  " + t[i][1].__str__() + "\n")
    ofile.close()
    firstPoint = t[0][1]
    secondPoint = t[1][1]
    firstBetaZero = firstPoint.label - np.asarray([firstPoint.x, firstPoint.y]).dot(beta)
    secondBetaZero = secondPoint.label - np.asarray([secondPoint.x, secondPoint.y]).dot(beta)
    print "beta0 = ", firstBetaZero
    print "beta0 = ", secondBetaZero
    betaZero = 0.5*(firstBetaZero + secondBetaZero)
    print "Percent error of beta0: ", abs(firstBetaZero - secondBetaZero)/firstBetaZero
    def curve(x, beta, betaZero, mu):
        return -x*beta[0]/beta[1] + (mu - betaZero)/beta[1]
    def generateBoundary(xLower, xUpper, beta, betaZero, mu, outputFileName):
        x = []
        y = []
        cutNumber = 20
        delta = (xUpper - xLower)/(float(cutNumber))
        for i in range(cutNumber+1):
            x.append(xLower + i*delta)
            y.append(curve(x[i], beta, betaZero, mu))
        ofile = open(outputFileName, "w")
        for i in range(len(x)):
            ofile.write(str(x[i]) + "  " + str(y[i]) + "\n")
        ofile.close()
    generateBoundary(xLower, xUpper, beta, betaZero, -1, "lowerBoundary.txt")
    generateBoundary(xLower, xUpper, beta, betaZero, 0, "boundary.txt")
    generateBoundary(xLower, xUpper, beta, betaZero, 1, "upperBoundary.txt")
    os.system("python pltfiles.py " + negativeFileName + "  " + positiveFileName + " lowerBoundary.txt boundary.txt upperBoundary.txt")

def main():
    import sys

    negativeFileName = "negativePoints.txt"
    positiveFileName = "positivePoints.txt"
    C = 1
    svm(negativeFileName, positiveFileName, C)

if __name__ == "__main__":
    import sys
    sys.exit(main())
