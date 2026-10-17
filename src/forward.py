import math

# ==========================
#    DATASET FORWARD PASS
# ==========================
#
# Hidden layer:
#   Calculates the weighted sum of inputs for each hidden neuron,
#   and produces ReLU activated hidden outputs.
#
# Output layer:
#   Calculates the weighted sum of hidden outputs,
#   and produces Sigmoid activated final probability.
#
# Returns:
#  - hidden_outputs   : Hidden layer outputs
#  - final_prediction : Output layer output

def relu(x):
    """ReLU activation function"""
    # Passes positive numbers through, turns negatives to 0."""
    return max(0.0, x)

def sigmoid(x):
    """Sigmoid activation function"""
    # Squashes any number into a range between 0.0 and 1.0.
    x = max(-500.0, min(500.0, x))      # Prevent math overflow if numbers get too large
    return 1.0 / (1.0 + math.exp(-x))   # math.exp(-x) = Epsilon to the power of -x

def forward_pass(x_inputs, W1, b1, W2, b2):
    """Passes an input vector through the network to get a final prediction."""
    
    # --- 1. Hidden Layer ---
    hidden_outputs = []
    for i in range(len(W1)):                        # For each hidden neuron:
        activation = 0.0
        for j in range(len(x_inputs)):              # For each input,
            activation += W1[i][j] * x_inputs[j]    # calculate the weighted sum: (w * x),
        activation += b1[i]                         # with bias,
        hidden_outputs.append(activation)           # and store the hidden output.

    # --- 2. Hidden Outputs ReLu Activation ---
    for i in range(len(hidden_outputs)):            # For each hidden output:
        hidden_outputs[i] = relu(hidden_outputs[i]) # Apply ReLU activation.    
        
    # --- 3. Output Layer ---
    output_activation = 0.0
    for i in range(len(hidden_outputs)):                  # For each hidden output:
        output_activation += W2[0][i] * hidden_outputs[i] # Calculate weighted sum from hidden outputs,
    output_activation += b2[0]                            # with bias, for the single output neuron.
    
    # --- 4. Output Layer Sigmoid Activation ---
    final_prediction = sigmoid(output_activation)   # Apply Sigmoid activation to get final probability (0 to 1)
    
    return hidden_outputs, final_prediction
