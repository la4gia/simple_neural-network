Simple Neural Networks

The bare bones of neural networks.
Simple layouts, without sci-kits, regularization, graphs, etc.
Hopefully helpful to understand backpropagation, with all the calculus and dimensionality of matrices fun.

-------------------------------------------------------------

both_true:

1 input layer with 2 units.
1 output layer with 1 unit.

Designed for two inputs, 0 or 1.
Outputs 1 if both inputs are 1, 0 for everything else.

-------------------------------------------------------------

all_or_none:

1 input layer with 2 units.
1 hidden layer with 2 units.
1 output layer with 1 unit.

Designed for two inputs, 0 or 1.
Outputs 1 if both inputs are 1 or 0, 0 for everything else.

-------------------------------------------------------------

mirror_array:

1 input layer with 2 units.
1 hidden layer with 3 units.
1 output layer with 4 units.

Designed for two inputs, 0 or 1.
00 = 1000, 01 or 10 = 0110, 11 = 0001.

-------------------------------------------------------------
