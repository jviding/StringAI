# String AI
Simple neural network created from a scratch without any libraries. Goal of this project 
is to learn the mathematical fundamentals of AI. This is to be done via exploration where
all the parameters are played around with to see how they influence performance and
predictions down to the per-neuron level.

The AI engine is stored under *src* and performance metrics under *lib*.


## Project structure
This project constitutes of:
- Generator: Creates a dataset for training
- Network: Creates neurons with weights and biases
- Forward: Generates a prediction
- Backward: Rebalances the neurons

Run a full cycle from dataset generation to training to testing:
> python3 mini.py


## Observations: Dataset
If the dataset doesn't have a balanced number of samples with and without Target, the network 
will take a lazy shortcut and start predicting either 1 or 2 for everything. This can be 
observed in the overall loss dropping very low immediately across the dataset.
```
Start training for 200 epochs...
Epoch 20/200 | Average Loss: 0.01940
Epoch 40/200 | Average Loss: 0.01930
Epoch 60/200 | Average Loss: 0.01928

--- Testing Network Predictions After Training ---
String: 'e' | Target: 1.0 | Prediction: 0.0683
String: 'i' | Target: 0.0 | Prediction: 0.0441
```

Whereas with approximately 50% balance across the dataset:
```
Start training for 200 epochs...
Epoch 20/200 | Average Loss: 0.04723
Epoch 40/200 | Average Loss: 0.04415
Epoch 60/200 | Average Loss: 0.04195

--- Testing Network Predictions After Training ---
String: 'e' | Target: 1.0 | Prediction: 0.9440
String: 'l' | Target: 0.0 | Prediction: 0.0000
```
