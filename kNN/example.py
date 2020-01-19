# example 3.1

def get_distance(x1, x2, p):
    summary = 0
    for pair in list(zip(x1, x2)):
        summary += pow(abs(pair[0] - pair[1]), p)
    return pow(summary, 1/p)

x1 = [1, 1]
x2 = [4, 4]
print(get_distance(x1, x2, 3))