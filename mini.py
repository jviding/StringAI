from src.generator import generate_dataset
from src.network import generate_layer_weights_and_biases
from src.forward import forward_pass
from src.backward import backward_pass, compute_loss

from lib.neurons import save_snapshot_of_W1_b1, plot_neuron_curves

# ===============================
# STEP 1: SETUP DATASET & NETWORK
# ===============================
DATASET_SIZE = 100    # Number of inputs
INPUT_SIZE   = 1      # Length of an input
TARGET       = "e"    # Target to identify from input

X_train, y_train, z_strs = generate_dataset(DATASET_SIZE, INPUT_SIZE, TARGET)

HIDDEN_SIZE  = 8      # Number of neurons for the hidden layer (a design choice)
OUTPUT_SIZE  = 1      # Only 1 final answer (a single probability of whether TARGET is present)

W1, b1, W2, b2 = generate_layer_weights_and_biases(OUTPUT_SIZE, HIDDEN_SIZE, INPUT_SIZE)

# --- Save a copy before training ---
save_snapshot_of_W1_b1(W1, b1, W2, b2)

# =========================================
# STEP 2: TRAINING LOOP (EPOCHS & BACKDROP)
# =========================================
EPOCHS        = 200
LEARNING_RATE = 0.1   # Magnitude of error correction in a forward pass

print(f"Start training for {EPOCHS} epochs...")

for epoch in range(EPOCHS):
    total_epoch_loss = 0.0
    
    for i in range(len(X_train)):
        sample = X_train[i]
        target = y_train[i][0]
        # 1. Forward Pass
        h_out, prediction = forward_pass(sample, W1, b1, W2, b2)
        # 2. Compute Loss
        loss = compute_loss(prediction, target)                     
        total_epoch_loss += loss
        # 3. Backward Pass & Weight Updates (in-place)
        backward_pass(sample, h_out, prediction, target, W1, b1, W2, b2, LEARNING_RATE)
        
    # Print average loss every 20 epochs to monitor learning progress
    if (epoch + 1) % 20 == 0:
        avg_loss = total_epoch_loss / DATASET_SIZE
        print(f"Epoch {epoch + 1}/{EPOCHS} | Average Loss: {avg_loss:.5f}")

# ====================================
# STEP 4: EVALUATION & PREDICTION TEST
# ====================================
print("\n--- Testing Network Predictions After Training ---")

for i in range(10):
    sample = X_train[i]
    target = y_train[i][0]
    
    # Run a final forward pass without updating weights
    h_out, prediction = forward_pass(sample, W1, b1, W2, b2)
    
    print(f"String: '{z_strs[i]}' | Target: {target} | Prediction: {prediction:.4f}")

# ================================
# STEP 5: PLOTTING THE TRANSITIONS
# ================================
save_snapshot_of_W1_b1(W1, b1, W2, b2)
plot_neuron_curves(HIDDEN_SIZE)
