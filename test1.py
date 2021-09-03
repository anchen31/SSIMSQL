import math

def myround(x, base=5):

    return base * math.floor(x/base)

print(myround(16))