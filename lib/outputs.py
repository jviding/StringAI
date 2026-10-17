from lib.storage import get_neurons_snapshots
import matplotlib.pyplot as plt


def get_y(phase, index, x_range):
    W1_snapshots, b1_snapshots, W2_snapshots, b2_snapshots = get_neurons_snapshots()

    # Calculate neuron output across input range
    W1 = W1_snapshots[phase][index][0]
    b1 = b1_snapshots[phase][index]
    W2 = W2_snapshots[phase][0][index]
    b2 = b2_snapshots[phase][0]
    return [max(0.0, (((W1 * x) + b1) * W2) + b2) for x in x_range]

def plot_output_curves(hidden_size):
    # Generate a range of inputs from 0.0 to 1.0
    x_range = [i / 100.0 for i in range(101)]

    fig, axes = plt.subplots(2, 4, figsize=(15, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    for i in range(hidden_size):
        y_before = get_y(0, i, x_range)
        y_mid = get_y(1, i, x_range)
        y_after = get_y(2, i, x_range)
        
        # Plotting
        ax = axes[i]
        ax.plot(x_range, y_before, color='gray', linestyle='--', label='Before Training')
        ax.plot(x_range, y_mid, color='black', linestyle='--', label='25% Trained')
        ax.plot(x_range, y_after, color='dodgerblue', linewidth=2, label='After Training')
        
        ax.set_title(f"Output Layer {i+1}")
        ax.set_xlim(0.0, 1.0)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)

    fig.suptitle("Output Layer Response Curves: Before vs. After Training", fontsize=14)
    plt.tight_layout()
    plt.savefig("assets/output2_boundary.png", dpi=300, bbox_inches="tight")
    plt.show()
