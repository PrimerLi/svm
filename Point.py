class Point:
    def __init__(self, x, y, label):
        self.x = x
        self.y = y
        self.label = label
    def __str__(self):
        return str(self.x) + "  " + str(self.y) + "  "+ str(self.label)

def inner(pointA, pointB):
    return pointA.x * pointB.x + pointA.y * pointB.y

def main():
    point = Point(1, 1, -1)
    print point
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
