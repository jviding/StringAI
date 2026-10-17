import random

# ===============================
#    DESIGN NETWORK DIMENSIONS
# ===============================
#
# Weight matrices will generated with random values from -0.5 to 0.5.
# Randomness breaks symmetry and ensures the neurons start with a unique perspective.
# Starting with small values prevents saturation in the activation functions.
#
# Returns:
#  - W1 : Weights for hidden neurons
#  - b1 : Biases for hidden neurons
#  - W2 : Weights for output neurons
#  - b2 : Biases for output neurons

def initialize_hidden_layer_weights_and_biases(hidden_size, input_size):
    """Generate a weight matrix with dimensions [hidden]x[input]."""
    # Each row represents a hidden neuron's personal set of weights.
    # Where these weights are then to be then matched against the input values.
    W1 = [[random.uniform(-0.5, 0.5) for _ in range(input_size)] for _ in range(hidden_size)]
    
    """Create biases for the hidden neurons."""
    # Biases almost always start at zero, so creates a list of zeros ([0.0. 0.0, ..., 0.0]).
    b1 = [0.0 for _ in range(hidden_size)]

    return W1, b1

def initialize_output_layer_weights_and_biases(output_size, hidden_size):
    """Generate a weight matrix with dimensions [output]x[hidden]."""
    # Neurons on the output layer receive input from all the hidden neurons.
    # So, the number of weights needs to be equal to the number of hidden neurons.
    W2 = [[random.uniform(-0.5, 0.5) for _ in range(hidden_size)] for _ in range(output_size)]

    """Create biases for the final output neuron."""
    # These will act as the starting bias.
    b2 = [0.0 for _ in range(output_size)]

    return W2, b2

def generate_layer_weights_and_biases(output_size, hidden_size, input_size):
    """Lock the starting seed to 42."""
    # Every time the script runs it will have the exact same starting weights. 
    # Reproducible results will be important for debugging.
    random.seed(42)

    W1, b1 = initialize_hidden_layer_weights_and_biases(hidden_size, input_size)
    W2, b2 = initialize_output_layer_weights_and_biases(output_size, hidden_size)

    return W1, b1, W2, b2
