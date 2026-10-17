from lib.storage import get_neurons_snapshots
import matplotlib.pyplot as plt
import math

def sigmoid(z):
    # Prevent overflow for large negative values
    if z < -709:
        return 0.0
    if z > 709:
        return 1.0
    return 1.0 / (1.0 + math.exp(-z))

def get_y(phase, hidden_size, x_range):
    W1_snapshots, b1_snapshots, W2_snapshots, b2_snapshots = get_neurons_snapshots()
    
    y_values = []
    for x in x_range:
        hidden_sum = 0.0
        for i in range(hidden_size):
            W1 = W1_snapshots[phase][i][0]
            b1 = b1_snapshots[phase][i]
            W2 = W2_snapshots[phase][0][i]
            
            # 1. Hidden layer: ReLU activation
            h = max(0.0, (W1 * x) + b1)
            # 2. Accumulate weighted hidden outputs
            hidden_sum += h * W2
            
        # 3. Output layer bias
        b2 = b2_snapshots[phase][0]
        pre_activation = hidden_sum + b2
        
        # 4. Final output activation (Sigmoid for classification)
        prediction = sigmoid(pre_activation)
        y_values.append(prediction)
        
    return y_values

def plot_prediction_curve(hidden_size):
    # Generate a range of inputs from 0.0 to 1.0
    x_range = [i / 100.0 for i in range(101)]

    fig, ax = plt.subplots(figsize=(10, 6))

    y_before = get_y(0, hidden_size, x_range)
    y_mid = get_y(1, hidden_size, x_range)
    y_after = get_y(2, hidden_size, x_range)
        
    # Plotting
    ax.plot(x_range, y_before, color='gray', linestyle='--', label='Before Training')
    ax.plot(x_range, y_mid, color='black', linestyle='--', label='25% Trained')
    ax.plot(x_range, y_after, color='dodgerblue', linewidth=2, label='After Training')
    
    ax.set_title("Network Decision Boundary (Final Prediction)")
    ax.set_xlabel("Input Range (x)")
    ax.set_ylabel("Probability / Output")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    fig.suptitle("Neural Network Evolution During Training", fontsize=14)
    plt.tight_layout()
    #plt.savefig("assets/prediction3_boundary.png", dpi=300, bbox_inches="tight")
    plt.show()
