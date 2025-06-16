import time
import cupy as cp
from os import urandom

from ciphers.gpu import present80_gpu
from ciphers.gpu import p
def make_train_data(encryption_function, plain_bits, key_bits, n, nr, delta_state=0, delta_key=0):
    """Sinh dữ liệu train cho NDC dùng cupy (GPU)."""
    # random keys & plaintext (0/1), tạo trên GPU
    keys0 = (cp.random.randint(0, 2, (n, key_bits), dtype=cp.uint8))
    pt0   = (cp.random.randint(0, 2, (n, plain_bits), dtype=cp.uint8))

    if isinstance(delta_key, int):
        delta_key = cp.zeros((n, key_bits), dtype=cp.uint8) + delta_key
    else:
        delta_key = cp.array(delta_key, dtype=cp.uint8)
    if isinstance(delta_state, int):
        delta_state = cp.zeros((n, plain_bits), dtype=cp.uint8) + delta_state
    else:
        delta_state = cp.array(delta_state, dtype=cp.uint8)

    keys1 = keys0 ^ delta_key
    pt1   = pt0 ^ delta_state

    # Encrypt song song trên GPU
    C0 = encryption_function(pt0, keys0, nr)
    C1 = encryption_function(pt1, keys1, nr)
    C = cp.hstack([C0, C1])

    # Sinh nhãn ngẫu nhiên Y trên GPU
    Y = cp.random.randint(0, 2, (n,), dtype=cp.uint8)
    num_rand_samples = int(cp.sum(Y == 0).get())

    if num_rand_samples > 0:
        # random ciphertext thay thế khi Y==0 (random sample)
        num_c = C0.shape[1] * 2  # Số bit mỗi dòng ciphertext
        rand_c = cp.random.randint(0, 2, (num_rand_samples, num_c), dtype=cp.uint8)
        C[Y == 0] = rand_c
    return C, Y

def integer_to_binary_array(int_val, num_bits):
    return cp.array([int(i) for i in bin(int_val)[2:].zfill(num_bits)], dtype = cp.uint8).reshape(1, num_bits)

def benchmark_make_train_data_gpu(encryption_func_gpu, n, rounds, plain_bits, key_bits):
    import cupy as cp
    cp.cuda.Stream.null.synchronize()
    start = time.time()
    C, Y = make_train_data(encryption_func_gpu, plain_bits, key_bits, n, rounds)
    cp.cuda.Stream.null.synchronize()
    end = time.time()
    print(f"[GPU/CuPy] Tạo {n} mẫu: {end-start:.4f}s ({n/(end-start):,.2f} mẫu/giây)")
    return C, Y

if __name__ == "__main__":
    encryption_function = present80_gpu.encrypt
    plain_bits = 64;
    key_bits =80
    num_samples = 1_000_000
    nr=32

    delta = integer_to_binary_array(0x90000000000, plain_bits)
    delta_key = 0
    delta_plain = delta[:plain_bits]

    print("hello world")

    data_generator = lambda num_samples, nr: make_train_data(
        encryption_function, plain_bits, key_bits, num_samples, nr, delta_plain, delta_key
    )

    C_gpu, Y_gpu = benchmark_make_train_data_gpu(make_train_data, encryption_function, num_samples, nr, plain_bits, key_bits)
    print(C_gpu.shape)