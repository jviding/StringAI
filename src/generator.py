import random
import string

# ================================
#    TRAINING DATASET GENERATOR
# ================================
# Returns:
#  - X_train : Encoded input Strings
#  - y_train : Targets
#  - z_strs  : Original input Strings

def generate_balanced_sample(length, target):
    # OBSERVATION: Target should be present in ~50% of samples
    has_target = random.random() < 0.5

    while True:
        s = generate_random_string(length, target)
        if has_target and target in s:
            return s
        elif not has_target and target not in s:
            return s

def generate_random_string(length, target):
    """Generate a random string of letters [a-z]."""
    # [a-z] = Hex [0x61-0x7A] = Dec [97-122]
    return ''.join(random.choices(string.ascii_lowercase, k=length))

def encode_string(s):
    """Encode by shifting and scaling to cover only the used ASCII range."""
    # Map values from 'a' (0.0) through 'z' (1.0)
    return [round((ord(c) - 97) / (122 - 97), 2) for c in s]

def get_target(s, target):
    """Return 1 if Target is present, otherwise 0."""
    return 1.0 if (target in s) else 0.0

def generate_dataset(samples, length, target):
    """Generate a dataset using lists of lists."""
    X_train = [] # Encoded Strings
    y_train = [] # If TARGET present
    z_strs  = [] # Original Strings
    for _ in range(samples):
        s = generate_balanced_sample(length, target)
        X_train.append(encode_string(s))
        y_train.append([get_target(s, target)])  # Nested list to match the single output structure
        z_strs.append(s)
        
    return X_train, y_train, z_strs
