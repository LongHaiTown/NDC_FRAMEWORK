from os import urandom
import os
import time
import numpy as np

from ciphers.cpu import present80
from ciphers.cpu import katan

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
    Y = np.frombuffer(urandom(n), dtype=np.uint8); Y = Y & 1;
    num_rand_samples = np.sum(Y==0);
    C[Y==0] = (np.frombuffer(urandom(num_rand_samples*C0.shape[1]*2),dtype=np.uint8)&1).reshape(num_rand_samples, -1)
    # Sanity check
    return C, Y

def benchmark_make_train_data_cpu(encryption_func_cpu, n, rounds, plain_bits, key_bits):
    start = time.time()
    C, Y = make_train_data(encryption_func_cpu, plain_bits, key_bits, n, rounds)
    end = time.time()
    print(f"[CPU/NumPy] Tạo {n} mẫu: {end-start:.4f}s ({n/(end-start):,.2f} mẫu/giây)")
    return C, Y

def save_make_ciphertextpair_train_data_cpu(encryption_func_cpu, n, rounds, plain_bits, key_bits, cipher_name, file_name="dataset_ciphertextpair_{cipher_name}.npz"):
    """Save X (n, plain_bits*3) and Y (n,) to a .npz file with cipher-specific naming."""
    save_path = os.path.join("../../data", file_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    C, Y = make_train_data(encryption_func_cpu, plain_bits, key_bits, n, rounds)
    np.savez_compressed(save_path, X=C, Y=Y)
    print(f"✅ Saved {cipher_name} dataset to {save_path}")
    return C,Y

def save_ciphertexpair_dataset_npz(X, Y, cipher_name,num_rounds,num_samples, save_path="{cipher_name}_ciphertextpair_{cipher_name}_round{num_rounds}_samples{num_samples}.npz"):
    """Save X (n, plain_bits*3) and Y (n,) to a .npz file with cipher-specific naming."""
    save_path = save_path.format(cipher_name=cipher_name,num_rounds=num_rounds,num_samples= num_samples)
    save_path = os.path.join("../../data", save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez_compressed(save_path, X=X, Y=Y)
    print(f"✅ Saved {cipher_name} dataset to {save_path}")

def integer_to_binary_array(int_val, num_bits):
    return np.array([int(i) for i in bin(int_val)[2:].zfill(num_bits)], dtype = np.uint8).reshape(1, num_bits)  

if __name__ == "__main__":
    encryption_function = present80.encrypt
    plain_bits = present80.plain_bits
    key_bits = present80.key_bits
    num_samples = 1_000_000
    nr = 32

    delta = integer_to_binary_array(0x90000000000, plain_bits)
    delta_key = 0
    delta_plain = delta[:plain_bits]

    data_generator = lambda num_samples, nr: make_train_data(
        encryption_function, plain_bits, key_bits, num_samples, nr, delta_plain, delta_key
    )

    # C_gpu, Y_gpu = benchmark_make_train_data_cpu(make_train_data, encryption_function, num_samples, nr, plain_bits, key_bits)
    # print(C_gpu.shape)

    C_gpu, Y_gpu = save_make_ciphertextpair_train_data_cpu(encryption_function, num_samples, nr, plain_bits, key_bits)

