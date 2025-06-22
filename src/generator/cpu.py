from os import urandom
import time
import numpy as np
import os
import tracemalloc
import time

def integer_to_binary_array_cpu(int_val, num_bits):
    return np.array([int(i) for i in bin(int_val)[2:].zfill(num_bits)], dtype = np.uint8).reshape(1, num_bits)

def make_train_data_cpu(encryption_function, plain_bits, key_bits, n, nr, delta_state=0, delta_key=0):
    # Generate keys and plaintexts
    keys0 = (np.frombuffer(urandom(n * key_bits), dtype=np.uint8) & 1).reshape(n, key_bits)
    pt0 = (np.frombuffer(urandom(n * plain_bits), dtype=np.uint8) & 1).reshape(n, plain_bits)

    # Apply deltas
    keys1 = keys0 ^ delta_key
    pt1 = pt0 ^ delta_state

    # Encrypt data
    C0 = encryption_function(pt0, keys0, nr)
    C1 = encryption_function(pt1, keys1, nr)
    C = np.hstack([C0, C1])

    # Generate labels
    Y = np.frombuffer(urandom(n), dtype=np.uint8) & 1
    num_rand_samples = np.sum(Y == 0)
    C[Y == 0] = (np.frombuffer(urandom(num_rand_samples * C0.shape[1] * 2), dtype=np.uint8) & 1).reshape(num_rand_samples, -1)

    # Return generated data and labels
    return C, Y
def save_sequence_dataset_npz(X, Y, cipher_name, save_path="dataset_sequence_{cipher_name}.npz"):
    """Save X (n, plain_bits*3) and Y (n,) to a .npz file with cipher-specific naming."""
    save_path = save_path.format(cipher_name=cipher_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez_compressed(save_path, X=X, Y=Y)
    print(f"✅ Saved {cipher_name} dataset to {save_path}")

def benchmark_make_train_data_cpu(encryption_function, n, rounds, plain_bits, key_bits, delta_state, delta_key):
    """Benchmark dataset generation."""
    # Start measuring time and memory
    tracemalloc.start()
    start = time.time()

    # Generate training data
    C, Y = make_train_data_cpu(encryption_function, plain_bits, key_bits, n, rounds, delta_state, delta_key)

    # End measuring time and memory
    end = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Log results
    execution_time = end - start
    samples_per_second = n / execution_time
    print(f"[CPU/NumPy] Generated {n} samples: {execution_time:.4f}s ({samples_per_second:,.2f} samples/second)")
    print(f"Execution time: {execution_time:.4f}s")
    print(f"Memory used: {current / 1024:.2f} KB \nPeak memory: {peak / 1024:.2f} KB")

    return C, Y

def generate_sequence_datasets_by_round(cipher_name, encryption_function, plain_bits, key_bits, rounds_range, num_samples, delta_p_hex, delta_key ,save_dir="../../data/"):
    """Generate and save datasets for multiple rounds for the specified cipher."""
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n🚀 Generating {cipher_name} sequence datasets for rounds {list(rounds_range)}")

    for r in rounds_range:
        print(f"\n📦 Generating {cipher_name} round {r} by CPU...")
        C, Y = benchmark_make_train_data_cpu(
            encryption_function,
            n=num_samples,
            rounds=r,
            plain_bits=plain_bits,
            key_bits=key_bits,
            delta_state=delta_p_hex,
            delta_key=delta_key
        )
        # save_path = os.path.join(save_dir, f"{cipher_name}_seq_round{r}_samples{num_samples}.npz")
        # save_sequence_dataset_npz(C, Y, cipher_name, save_path=save_path)
    print(f"\n🏁 All {cipher_name} datasets generated and saved.")


def benchmark_encryption_cpu(encryption_function,batch_size, num_round):
    p = np.zeros((batch_size, 64), dtype=np.uint8)
    k = np.zeros((batch_size, 80), dtype=np.uint8)

    tracemalloc.start()
    start = time.time()

    _ = encryption_function(p, k, num_round)

    end = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    execution_time = end - start
    speed = batch_size / execution_time
    print(f"{batch_size} blocks, {num_round} rounds:")
    print(f"Execution time: {execution_time:.4f} seconds")
    print(f"Speed: {speed:,.2f} blocks/sec")
    print(f"Memory used: {current / 1024:.2f} KB")
    print(f"Peak memory: {peak / 1024:.2f} KB")

if __name__ == "__main__":
    cipher_name = "present"  
    encryption_function = present80.encrypt
    plain_bits = present80.plain_bits
    key_bits = present80.key_bits
    num_samples = 1_000_000
    rounds_range = [10]
    delta_p_hex = "0x90"
    delta = integer_to_binary_array_cpu(int(delta_p_hex, 16), plain_bits)
    delta_plain = delta[:plain_bits]
    delta_key = 0
    # print(f"Benchmarking encryption for {cipher_name}...")
    # benchmark_encryption_cpu(encryption_function,num_samples, num_round = 11)
    # generate_sequence_datasets_by_round(
    #     cipher_name=cipher_name,
    #     encryption_function=encryption_function,
    #     plain_bits=plain_bits,
    #     key_bits=key_bits,
    #     rounds_range=rounds_range,
    #     num_samples=num_samples,
    #     delta_p_hex=delta_plain,
    #     delta_key=delta_key
    # )