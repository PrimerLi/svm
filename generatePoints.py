#!/usr/bin/env python

import Point
import random

def printPoints(points, outputFileName):
    ofile = open(outputFileName, "w")
    for i in range(len(points)):
        ofile.write(points[i].__str__() + "\n")
    ofile.close()

def main():
    import os
    import sys

    if (len(sys.argv) != 3):
        print "n = sys.argv[1], p = sys.argv[2]. "
        return -1

    negativePoints = []
    positivePoints = []
    numberOfNegatives = int(sys.argv[1])
    numberOfPositives = int(sys.argv[2])

    for i in range(numberOfNegatives):
        randomX = random.uniform(-1, 1)
        randomY = random.uniform(-1, 2)
        point = Point.Point(randomX, randomY, -1)
        negativePoints.append(point)
    for i in range(numberOfPositives):
        randomX = 3 + random.uniform(-1.4, 2)
        randomY = 2 + random.uniform(-1, 1)
        point = Point.Point(randomX, randomY, 1)
        positivePoints.append(point)

    printPoints(negativePoints, "negativePoints.txt")
    printPoints(positivePoints, "positivePoints.txt")

    #os.system("./pltfiles.py negativePoints.txt positivePoints.txt")
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
