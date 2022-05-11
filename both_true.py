import numpy as np
# np.seterr(divide='ignore')
# np.seterr(invalid='ignore')


# APPLY SIGMOID
def sig(z):
    return 1 / (1 + (np.e**(-z)))


# DEFINE Z
def z(x, w, b):
    # x (4, 2) @ w (2, 1) ---> (4, 1)
    return sig(np.dot(x, w) + b)


# DEFINE COST FUNCTION (TO MONITOR PROGRESS)
def cost(a, y, m):

    # COST FUNCTION IN ONE PIECE
    # cost = (-1 * ((y * np.log(a)) + ((1 - y) * np.log(1 - a)))) / m

    # ERROR WHEN Y == 1
    cost_class1 = -y * np.log(a)

    # ERROR WHEN Y == 0
    cost_class0 = (1 - y) * np.log(1 - a)

    # SUM OF BOTH ERRORS
    cost = cost_class1 - cost_class0

    # TAKE THE AVERAGE ERROR
    cost = cost.sum() / m

    return cost


# APPLY GRADIENT DESCENT (WHERE THE MAGIC HAPPENS)
def grad(a, x, y, w, b, m, lr):

    # a = (4, 1)
    # x = (4, 2)
    # y = (4, 1)
    # w = (2, 1)

    # dw = x.T (2, 4) @ [a - y] (4, 1) ---> (2, 1)

    # GET DERIAVATIVE OF WEIGHTS
    dw = np.dot(x.T, (a - y))
    dw /= m

    # GET DERIAVATIVE OF BIAS
    db = np.sum(a - y)
    db /= m

    # APPLY LEARNING RATE
    dw *= lr
    db *= lr

    # UPDATE WEIGHTS AND BIAS
    w = w - dw  # w (2, 1) - dw (2, 1)
    b = b - db

    return w, b


print(" ")
print("Both Trues. Outputs 1 if inputs are both 1. Outputs 0 for anything else")
print(" ")

# DEFINE INPUTS, OUTPUTS, WEIGHTS, BIAS, ROWS,
#                           GOAL, LEARNING RATE, & QUIZ ARRAY

# x = (4, 2) (m, f)
# y = (4, 1) (m, o)
# w = (2, 1) (f, o)

# DATA SAMPLE
x = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([0, 0, 0, 1]).reshape(4, 1)  # DESIRED OUTPUTS
w = np.zeros((2, 1))  # INITIAL WEIGHTS
b = 0  # INITIAL BIAS
m = np.size(y)  # TOTAL NUMBER OF ROWS
GOAL = 0.009  # COST GOAL TO END LEARNING LOOP
lr = .7  # LEARNING RATE

guess = np.array([0, 0])  # QUIZ ARRAY

# QUIZ BEFORE TRAINING
print(f"Initial answer for {guess}:", 1 if z(guess, w, b) >= .5 else 0)
print()
print("Learning...")
print()


j = 1  # STARTING COST
i = 0  # ITERATION COUNTER

# LEARNING LOOP
while j > GOAL:
    a = z(x, w, b)  # UPDATE ACTIVATION VARIABLE
    if i % 1000 == 0:  # PRINT COST VALUE AT SPECIFIED ITERATION
        print(j)
    w, b = grad(a, x, y, w, b, m, lr)  # ACTIVATE GRADIENT DESCENT
    j = cost(a, y, m)  # UPDATE COST VARIABLE (TO MONITOR PROGRESS)
    i += 1

print(j)  # PRINT FINAL COST

# PRINT FINAL WEIGHTS, BIAS, AND ITERATION COUNT
print(" ")
print(f'The minimum occurs at {i}')
print(f'The weight is now {w}')
print(f'The bias is now {b}\n')

# QUIZ AFTER TRAINING
print(f"New answer for {guess}:", 1 if z(guess, w, b) >= .5 else 0)
print()
