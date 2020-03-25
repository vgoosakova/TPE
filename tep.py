import numpy as np

a0 = 1
a1 = 2
a2 = 3
a3 = 4


def matrix1(start, end):
    return np.random.randint(start, end, (8, 3))


a = matrix1(1, 21)


min = np.amin(a, axis=0)
max = np.amax(a, axis=0)

b = np.concatenate((a, np.zeros([2, 3])))
c = np.concatenate((b, np.zeros([10, 4])), axis=1)

#  x0
for i in range(3):
    c[8][i] = (min[i]+max[i])/2

#  dx
for i in range(3):
    c[9][i] = (c[8][i] - min[i])

#  y(xi)
for i in range(0, 8):
    c[i][3] = a0+a1*c[i][0]+a2*c[i][1]+a3*c[i][2]

# xi(n)
for i in range(8):
    c[i][4] = ((c[i][0] - c[8][0]) / (c[9][0]))
    c[i][5] = ((c[i][1] - c[8][1]) / (c[9][1]))
    c[i][6] = ((c[i][2] - c[8][2]) / (c[9][2]))

print("\n")
print("  x1    x2    x3    y   x1(n)  x2(n)  x3(n)" )

for i in range(10):
    for j in range(7):
        print("{:>5.1f}".format(c[i][j]), end=" ")
    print("\t")

print("---------------------------------------------")

# y_et
y_et = a0+a1*c[8][0]+a2*c[8][1]+a3*c[8][2]
print("Y_et = ", y_et)

y = []
for i in range(8):
    y.append((c[i][3]-y_et)*(c[i][3]-y_et))

y = np.amin(y)
print("Y = ", y)

