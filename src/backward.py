# ==========================
#   LOSS AND BACKPROPAGATION
# ==========================
#
# 1. Produce delta2 that is the error multiplied by confidence. If the output 
#    neuron is confident (prediction closer to 1 or 0), the error is diluted.
#
# 2. Use delta2 to update the output weights and bias (according to learning_rate).
#
# 3. Produce delta1 that is the error multiplied by the connecting weight of the hidden
#    neuron. The more weight a neuron had in the prediction, the more its correction.
#
# 4. Use delta1 to update the hidden weights and bias (according to learning_rate).
#    If relu_derivative was 0, the neuron is dead and won't be updated.

def compute_loss(prediction, target):
    """Calculate Mean Squared Error (MSE) loss"""
    # 0.5 * (prediction - target)^2
    return 0.5 * (prediction - target) ** 2

def backward_pass(x_inputs, hidden_outputs, prediction, target, W1, b1, W2, b2, learning_rate):
    """Perform backpropagation"""
    # Use the chain rule, calculate gradients, 
    # and update weights and biases in-place.
    
    # --- 1. Output Layer Gradients ---
    # Derivative of MSE loss w.r.t prediction: (prediction - target)
    # Derivative of Sigmoid: prediction * (1.0 - prediction)
    error = prediction - target
    sigmoid_derivative = prediction * (1.0 - prediction)
    delta2 = error * sigmoid_derivative  # Combined error signal for output layer

    # --- 2. Update Output Weights (W2) and Bias (b2) ---
    # W2 has shape [1][hidden_size]
    for i in range(len(W2[0])):
        grad_w2 = delta2 * hidden_outputs[i]
        W2[0][i] -= learning_rate * grad_w2
    
    b2[0] -= learning_rate * delta2

    # --- 3. Hidden Layer Gradients ---
    # Propagate the error backward from output to hidden neurons
    delta1 = []
    for i in range(len(hidden_outputs)):
        # ReLU derivative: 1 if the post-activation output > 0, else 0
        relu_derivative = 1.0 if hidden_outputs[i] > 0 else 0.0
        
        # Error signal for hidden neuron i = (delta2 * connecting weight) * relu_derivative
        d1 = delta2 * W2[0][i] * relu_derivative
        delta1.append(d1)

    # --- 4. Update Hidden Weights (W1) and Biases (b1) ---
    # W1 has shape [hidden_size][input_size]
    for i in range(len(W1)):
        for j in range(len(x_inputs)):
            grad_w1 = delta1[i] * x_inputs[j]
            W1[i][j] -= learning_rate * grad_w1
        b1[i] -= learning_rate * delta1[i]
