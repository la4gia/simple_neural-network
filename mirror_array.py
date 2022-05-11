import numpy as np


class go_dogs_go:
    def __init__(self, t):
        input_units = 2  # UNITS IN INPUT LAYER
        hidden_units = 3  # UNITS IN HIDDEN LAYER
        output_units = 4  # UNITS IN OUTPUT LAYER

        # SEPARATE TRAINING DATA INTO FEATURES AND DESIRED 'Y'
        features = t[:, 0]
        self.features = np.array([[features[:][0]],
                                  [features[:][1]],
                                  [features[:][2]],
                                  [features[:][3]]], dtype=float).reshape(4, 2)
        desires = t[:, 1]
        self.y = np.array([desires[:][0], desires[:][1],
                           desires[:][2], desires[:][3]], dtype=float)

        # RANDOMIZE WEIGHTS (w2 --> (2x3), w3 --> (3x4)) & BIAS
        self.w2 = np.random.randn(input_units, hidden_units)
        self.w3 = np.random.randn(hidden_units, output_units)
        self.b = np.zeros((1, 2))

        # VARIABLIZE # OF ROWS & LEARNING RATE
        self.m = len(self.y)
        self.lr = .03

    # APPLY SIGMOID
    def sig(self, feets, weights, bias):
        return 1 / (1 + np.e**(-(np.dot(feets, weights) + bias)))

    # APPLY SIGMOID PRIME
    def sig_dash(self, a):
        return a * (1 - a)

    # DEFINE COST FUNCTION (TO MONITOR PROGRESS)
    def cost(self):

        # COST FUNCTION IN ONE PIECE
        # error = (-1 * ((y * np.log(a)) + ((1 - y) * np.log(1 - a)))) / self.m

        error = []  # LIST TO GATHER EACH OUTPUT ERROR

        # LOOP THROUGH COST FOR EACH OUTPUT
        for i in range(self.m):

            # ERROR WHEN Y == 1
            error_class1 = -self.y[i] * np.log(self.a3[i])

            # ERROR WHEN Y == 0
            error_class0 = (1 - self.y[i]) * np.log(1 - self.a3[i])

            # SUM OF BOTH ERRORS
            error.append(sum(error_class1 - error_class0))

        return sum(error) / self.m  # AVERAGE ERROR

    # FORWARD PROPAGATION
    def forward_props(self, g=None):
        if g is not None:   # FOR QUIZ
            features = g
        else:
            features = self.features

        # *** NOTE: THE INPUTS ARE LAYER 1 (a1) *** #

        # ACTIVATE ON LAYER 2
        # features (4x2) @ w2 (2x3) ---> (4x3) "a2"
        self.a2 = self.sig(features, self.w2, self.b[0][0])

        # ACTIVATE ON LAYER 3
        # a2 (4x3) @ w3 (3x4) ---> (4x4) "a3"
        self.a3 = self.sig(self.a2, self.w3, self.b[0][1])

        return self.a3

    # BACK SCRATCHER
    def back_props(self):

        # DEFINE DELTA3
        # a3 (4x4) - y (4x4) ---> (4x4) "delta3"
        delta3 = (self.a3 - self.y)

        # GET DERIAVATIVE OF W3 & B3
        # a2.T (3x4) @ delta3 (4x4) ---> (3x4) "dw3"
        dw3 = np.dot(self.a2.T, delta3)
        dw3 / self.m
        db3 = delta3.sum() / self.m

        # DEFINE DELTA2
        # delta3 (4x4) @ w3.T (4x3) ---> (4x3) "power through"
        # a2 (4x3) * [1 - a2] (4x3) ---> (4x3) "sigdiv2"
        # power through(4x3) * sigdiv2 (4x3) ---> (4x3) "delta2"
        delta2 = np.multiply(
            np.dot(delta3, self.w3.T), self.sig_dash(self.a2))

        # GET DERIAVATIVE OF W2 & B2
        # features.T (2x4) @ delta2 (4x3) ---> (2x3) "dw2"
        dw2 = np.dot(self.features.T, delta2)
        dw2 / self.m
        db2 = delta2.sum() / self.m

        # APPLY LEARNING RATE
        dw3 *= self.lr
        dw2 *= self.lr
        db3 *= self.lr
        db2 *= self.lr

        # UPDATE WEIGHTS
        self.w3 -= dw3
        self.w2 -= dw2

        # UPDATE BIASES
        self.b[0][1] -= db3
        self.b[0][0] -= db2

        return

    # APPLY DECISION BOUNDARY (FOR QUIZ)
    def decision_boundary(self, prob):
        return 1 if prob >= 0.5 else 0

    # PREPARE VECTOR FOR DECISION BOUNDARY (FOR QUIZ)
    def flat_it(self, predictions):
        dbound = np.vectorize(self.decision_boundary)  # MAKES FUNCTION A LOOP
        return dbound(predictions).flatten()  # MAKES VECTOR ONE ROW


# GROUND ZERO
if __name__ == '__main__':

    print(" ")

    print("An input of 00 should equal an ouput of 1000")
    print("An input of 01 or 10 should equal an output of 0110")
    print("An input of 11 should equal an ouput of 0001")

    # TRAINING DATA ([(INPUTS, [DESIRED OUTPUTS])])
    trainer = np.array([([0, 0], [1, 0, 0, 0]),
                        ([0, 1], [0, 1, 1, 0]),
                        ([1, 0], [0, 1, 1, 0]),
                        ([1, 1], [0, 0, 0, 1])], dtype=object)

    gdg = go_dogs_go(trainer)  # AND THEY'RE OFF!

    # QUIZ ARRAYS
    guess_red = np.array([0, 0])
    guess_yellow0 = np.array([0, 1])
    guess_yellow1 = np.array([1, 0])
    guess_green = np.array([1, 1])

    # ANSWERS TO QUIZ
    red = np.array([1, 0, 0, 0])
    yellow = np.array([0, 1, 1, 0])
    green = np.array([0, 0, 0, 1])

    # QUIZ LIST
    quiz_list = [(guess_red, "guess_red"),
                 (guess_yellow0, "guess_yellow0"),
                 (guess_yellow1, "guess_yellow1"),
                 (guess_green, "guess_green")]

    # TEST BEFORE LEARNING
    print(" ")
    print(f"00 input = {gdg.flat_it(gdg.forward_props(guess_red))}")
    print(f"01 input = {gdg.flat_it(gdg.forward_props(guess_yellow0))}")
    print(f"10 input = {gdg.flat_it(gdg.forward_props(guess_yellow1))}")
    print(f"11 input = {gdg.flat_it(gdg.forward_props(guess_green))}")

    i = 0  # ITERATION COUNTER
    j = 1  # INITIAL COST VARIABLE
    GOAL = .09  # SET COST GOAL

    print()
    print("Learning...")
    print()

    # LEARNING LOOP
    while j > GOAL:
        gdg.forward_props()  # ACTIVATE FORWARD PROPAGATION
        j = gdg.cost()  # UPDATE COST VARIABLE
        if i % 100 == 0:
            print(j)  # PRINT COST AT SPECIFIED ITERATION (TO MONITOR PROGRESS)
        gdg.back_props()  # ACTIVATE BACKWARD PROPAGATION
        i += 1

    print(j)  # PRINT FINAL COST
    print()
    print("Goal reached after", i, "iterations")
    print(' ')

    # TEST AFTER LEARNING
    print(f"00 input = {gdg.flat_it(gdg.forward_props(guess_red))}")
    print(f"01 input = {gdg.flat_it(gdg.forward_props(guess_yellow0))}")
    print(f"10 input = {gdg.flat_it(gdg.forward_props(guess_yellow1))}")
    print(f"11 input = {gdg.flat_it(gdg.forward_props(guess_green))}")
    print()
