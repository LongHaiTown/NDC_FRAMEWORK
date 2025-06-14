from os import urandom
import time
import numpy as np
import os

from ciphers.cpu import lea, present80, tea, speck64128, simon128256, katan, hight, gimli
def integer_to_binary_array(int_val, num_bits):
    return np.array([int(i) for i in bin(int_val)[2:].zfill(num_bits)], dtype = np.uint8).reshape(1, num_bits)

# def make_train_data(encryption_function, plain_bits, key_bits, n, rounds, delta_state=0, delta_key=0):
    """
    Generate dataset for the specified cipher.
    Output: X shape (n, plain_bits*3), Y shape (n,)
    - X: Concatenation of C0, C1, C0^C1 (each plain_bits bits, total plain_bits*3 bits)
    - Y: Labels (1 if P1 = P0 ^ delta_state, 0 if P1 is random)
    """
    # Generate random keys (n samples, each key is key_bits bits)
    keys0 = np.random.randint(0, 2, (n, key_bits), dtype=np.uint8)
    keys1 = keys0 ^ delta_key  # Apply delta_key if provided

    # Generate random plaintexts (n samples, each plaintext is plain_bits bits)
    pt0 = np.random.randint(0, 2, (n, plain_bits), dtype=np.uint8)
    pt1 = pt0 ^ delta_state  # P1 = P0 ^ delta_state for related pairs

    # Generate labels
    Y = np.random.randint(0, 2, n, dtype=np.uint8)
    num_rand_samples = np.sum(Y == 0)

    # For random pairs (Y=0), replace pt1 with random plaintext
    if num_rand_samples > 0:
        pt1[Y == 0] = np.random.randint(0, 2, (num_rand_samples, plain_bits), dtype=np.uint8)

    # Encrypt
    C0 = encryption_function(pt0, keys0, rounds)  # Shape: (n, plain_bits)
    C1 = encryption_function(pt1, keys1, rounds)  # Shape: (n, plain_bits)
    C_XOR = np.bitwise_xor(C0, C1)  # Shape: (n, plain_bits)

    # Concatenate C0, C1, C0^C1
    X = np.hstack([C0, C1, C_XOR])  # Shape: (n, plain_bits*3)

    return X, Y

def make_train_data(encryption_function, plain_bits, key_bits, n, nr, delta_state=0, delta_key=0):
    """TEMPORARY VERSION."""
    keys0 = (np.frombuffer(urandom(n*key_bits),dtype=np.uint8)&1)
    keys0 = keys0.reshape(n, key_bits);
    pt0 = (np.frombuffer(urandom(n*plain_bits),dtype=np.uint8)&1).reshape(n, plain_bits);
    keys1 = keys0^delta_key
    pt1 = pt0^delta_state
    C0 = encryption_function(pt0, keys0, nr)
    C1 = encryption_function(pt1, keys1, nr)
    C = np.hstack([C0, C1])
    Y = np.frombuffer(urandom(n), dtype=np.uint8); Y = Y & 1
    num_rand_samples = np.sum(Y==0);
    C[Y==0] = (np.frombuffer(urandom(num_rand_samples*C0.shape[1]*2),dtype=np.uint8)&1).reshape(num_rand_samples, -1)
    # Sanity check
    return C, Y
def save_sequence_dataset_npz(X, Y, cipher_name, save_path="dataset_sequence_{cipher_name}.npz"):
    """Save X (n, plain_bits*3) and Y (n,) to a .npz file with cipher-specific naming."""
    save_path = save_path.format(cipher_name=cipher_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez_compressed(save_path, X=X, Y=Y)
    print(f"✅ Saved {cipher_name} dataset to {save_path}")

def benchmark_make_train_data(encryption_function, n, rounds, plain_bits, key_bits, delta_state=0, delta_key=0):
    """Benchmark dataset generation."""
    start = time.time()
    X, Y = make_train_data(encryption_function, plain_bits, key_bits, n, rounds, delta_state, delta_key)
    end = time.time()
    print(f"[CPU/NumPy] Generated {n} samples: {end-start:.4f}s ({n/(end-start):,.2f} samples/second)")
    return X, Y

def generate_sequence_datasets_by_round(cipher_name, encryption_function, plain_bits, key_bits, rounds_range, num_samples, delta_p_hex, save_dir="/data/"):
    """Generate and save datasets for multiple rounds for the specified cipher."""
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n🚀 Generating {cipher_name} sequence datasets for rounds {list(rounds_range)}")

    # Convert delta_p_hex to binary array
    delta = integer_to_binary_array(int(delta_p_hex, 16), plain_bits)
    delta_key = 0  

    for r in rounds_range:
        print(f"\n📦 Generating {cipher_name} round {r}...")
        X, Y = benchmark_make_train_data(
            encryption_function,
            n=num_samples,
            rounds=r,
            plain_bits=plain_bits,
            key_bits=key_bits,
            delta_state=delta,
            delta_key=delta_key
        )
        save_path = os.path.join(save_dir, f"{cipher_name}_seq_round{r}_samples{num_samples}.npz")
        save_sequence_dataset_npz(X, Y, cipher_name, save_path=save_path)
    print(f"\n🏁 All {cipher_name} datasets generated and saved.")

if __name__ == "__main__":
    cipher_name = "gimli"  
    encryption_function = gimli.encrypt
    plain_bits = gimli.plain_bits
    key_bits = gimli.key_bits
    num_samples = 1_000
    rounds_range = [10]
    delta_p_hex = "0x90"
    generate_sequence_datasets_by_round(
        cipher_name=cipher_name,
        encryption_function=encryption_function,
        plain_bits=plain_bits,
        key_bits=key_bits,
        rounds_range=rounds_range,
        num_samples=num_samples,
        delta_p_hex=delta_p_hex
    )