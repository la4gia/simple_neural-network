import numpy as np
# from professor import profile


class NN:
    def __init__(self, x, y, m, lr):
        # DEFINE UNITS IN LAYERS
        input_units = 2
        hidden_units = 2
        output_units = 1

        # VARIABLIZE (yep) INPUTS, OUTPUTS, # OF ROWS
        self.features = x  # (4x2)
        self.y = y  # (4x1)
        self.m = m

        # RANDOMIZE WEIGHTS (w1 --> (2x2), w2 --> (2x1)) & BIAS
        self.w1 = np.random.randn(input_units, hidden_units)
        self.w2 = np.random.randn(hidden_units, output_units)
        self.b = np.zeros((1, 3))  # SKIPPING [0][0] SO LAYERS LOOK PRETTY

        # INITIALIZE LEARING RATE (ALPHA)
        self.lr = lr

    # APPLY SIGMOID
    def sig(self, feets, weights, bias):
        return 1 / (1 + np.e**(-(np.dot(feets, weights) + bias)))

    # APPLY SIGMOID PRIME
    def sig_dash(self, a):
        return a * (1 - a)

    # DEFINE COST FUNCTION (TO MONITOR PROGRESS)
    def cost(self):

        # COST FUNCTION IN ONE PIECE
        # final_error = (-1 * ((y * np.log(a)) + ((1 - y) * np.log(1 - a)))) / self.m

        # ERROR WHEN Y == 1
        error_class1 = -self.y * np.log(self.a3)

        # ERROR WHEN Y == 0
        error_class0 = (1 - self.y) * np.log(1 - self.a3)

        # SUM OF BOTH ERRORS
        error = error_class1 - error_class0

        # TAKE THE AVERAGE ERROR
        final_error = error.sum() / self.m

        return final_error

    # FORWARD PROPAGATION
    def forward_props(self, g=None):
        if g is not None:   # FOR QUIZ
            features = g
        else:
            features = self.features

        # *** NOTE: THE INPUTS ARE LAYER 1 (a1) *** #

        # ACTIVATE ON LAYER 2
        # features (4x2) @ w1 (2x2) ---> (4x2) "a2"
        self.a2 = self.sig(features, self.w1, self.b[0][1])

        # ACTIVATE ON LAYER 3
        # a2 (4x2) @ w2 (2x1) ---> (4x1) "a3"
        self.a3 = self.sig(self.a2, self.w2, self.b[0][2])

        return self.a3

    # BACK PROPAGATION (Deep breath...)
    def back_props(self):

        # DEFINE DELTA3
        # a3 (4x1) - y (4x1) ---> (4x1) "delta3"
        delta3 = (self.a3 - self.y)

        # GET DERIAVATIVE OF W2 & B2
        # a2.T (2x4) @ delta3 (4x1) ---> (2x1) "dw2"
        dw2 = np.dot(self.a2.T, delta3)
        dw2 / self.m
        db2 = delta3.sum() / self.m

        # DEFINE DELTA2
        # delta3 (4x1) @ w2.T (1x2) ---> (4x2) "hubba hubba"
        # a2 (4x2) * [1 - a2] (4x2) ---> (4x2) "sigdiv2"
        # hubba hubba (4x2) * sigdiv2 (4x2) ---> (4x2) "delta2"
        delta2 = np.multiply(
            np.dot(delta3, self.w2.T), self.sig_dash(self.a2))

        # GET DERIAVATIVE OF W1 & B1
        # features.T (2x4) @ delta2 (4x2) ---> (2x2) "dw1"
        dw1 = np.dot(self.features.T, delta2)
        dw1 / self.m
        db1 = delta2.sum() / self.m

        # APPLY LEARNING RATE
        dw2 *= self.lr
        dw1 *= self.lr
        db2 *= self.lr
        db1 *= self.lr

        # UPDATE WEIGHTS
        self.w2 -= dw2
        self.w1 -= dw1

        # UPDATE BIASES
        self.b[0][2] -= db2
        self.b[0][1] -= db1

        return


# GROUND ZERO
if __name__ == '__main__':

    # GREETINGS! 
    print(' ')
    print('All Or None: Outputs 1 if inputs are the same. Outputs 0 if inputs or different.')

    # DEFINE INPUTS, OUTPUTS, # OF ROWS, learning rate (lr), GOAL
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).reshape(4, 2)
    y = np.array([1, 0, 0, 1]).reshape(4, 1)
    m = np.size(y)
    lr = .05
    GOAL = .099  # COST GOAL TO END LOOP

    guess1 = np.array([0, 0])  # QUIZ ARRAY
    guess2 = np.array([0, 1])  # QUIZ ARRAY
    guess3 = np.array([1, 0])  # QUIZ ARRAY
    guess4 = np.array([1, 1])  # QUIZ ARRAY

    nn = NN(x, y, m, lr)  # INITIATE NEURAL NETWORK

    # PRINT INITIAL WEIGHTS AND BIAS
    print(" ")
    print(f'The starting weight for layer 1: \n{nn.w1}')
    print(" ")
    print(f'The starting weight for layer 2: \n{nn.w2}')
    print(" ")
    print(f'The starting biases: {nn.b[0][1:]}\n')

    # QUIZ BEFORE TRAINING
    print('Initial answer for inputs (0,0):', 1 if nn.forward_props(guess1) >= 0.5 else 0)
    print('Initial answer for inputs (0,1):', 1 if nn.forward_props(guess2) >= 0.5 else 0)
    print('Initial answer for inputs (1,0):', 1 if nn.forward_props(guess3) >= 0.5 else 0)
    print('Initial answer for inputs (1,1):', 1 if nn.forward_props(guess4) >= 0.5 else 0)
    print()
    print("Learning...")
    print()

    i = 0  # ITERATION COUNTER
    j = 1  # STARTING COSTS

    # LEARNING LOOP
    while j > GOAL:
        nn.forward_props()  # FORWARD PROPAGATION
        j = nn.cost()  # UPDATE COST VARIABLE (TO MONITOR PROGRESS)
        if i % 1000 == 0:  # PRINT COST AT SPECIFIED ITERATION
            print(j)
        nn.back_props()  # BACK PROPAGATION
        i += 1

    print(j)  # FINAL COST

    # PRINT FINAL WEIGHTS, BIAS, AND ITERATION COUNT
    print(" ")
    print(f'The minimum occurs at {i}')
    print(" ")
    print(f'The final weight for layer 1: \n{nn.w1}')
    print(" ")
    print(f'The final weight for layer 2: \n{nn.w2}')
    print(" ")
    print(f'The final biases: {nn.b[0][1:]}\n')

    # QUIZ AFTER TRAINING
    print('New answer for inputs (0,0):', 1 if nn.forward_props(guess1) >= 0.5 else 0)  # 1
    print('New answer for inputs (0,1):', 1 if nn.forward_props(guess2) >= 0.5 else 0)  # 0
    print('New answer for inputs (1,0):', 1 if nn.forward_props(guess3) >= 0.5 else 0)  # 0
    print('New answer for inputs (1,1):', 1 if nn.forward_props(guess4) >= 0.5 else 0)  # 1
    print()
