import cupy as cp
import os
import tracemalloc
import time
import pynvml

def integer_to_binary_array_gpu(int_val, num_bits):
    return cp.array([int(i) for i in bin(int_val)[2:].zfill(num_bits)], dtype=cp.uint8).reshape(1, num_bits)

def make_train_data_gpu(encryption_function, plain_bits, key_bits, n, nr, delta_state=0, delta_key=0):
    """Sinh dữ liệu train cho NDC dùng cupy (GPU)."""
    keys0 = cp.random.randint(0, 2, (n, key_bits), dtype=cp.uint8)
    pt0 = cp.random.randint(0, 2, (n, plain_bits), dtype=cp.uint8)

    if isinstance(delta_key, int):
        delta_key = cp.zeros((n, key_bits), dtype=cp.uint8) + delta_key
    else:
        delta_key = cp.array(delta_key, dtype=cp.uint8)
    if isinstance(delta_state, int):
        delta_state = cp.zeros((n, plain_bits), dtype=cp.uint8) + delta_state
    else:
        delta_state = cp.array(delta_state, dtype=cp.uint8)

    keys1 = keys0 ^ delta_key
    pt1 = pt0 ^ delta_state

    C0 = encryption_function(pt0, keys0, nr)
    C1 = encryption_function(pt1, keys1, nr)
    C = cp.hstack([C0, C1])

    Y = cp.random.randint(0, 2, (n,), dtype=cp.uint8)
    num_rand_samples = int(cp.sum(Y == 0).get())

    if num_rand_samples > 0:
        num_c = C0.shape[1] * 2
        rand_c = cp.random.randint(0, 2, (num_rand_samples, num_c), dtype=cp.uint8)
        C[Y == 0] = rand_c

    return C, Y

def save_sequence_dataset_npz(X, Y, cipher_name, save_path="dataset_sequence_{cipher_name}.npz"):
    save_path = save_path.format(cipher_name=cipher_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cp.savez_compressed(save_path, X=X, Y=Y)
    print(f"✅ Saved {cipher_name} dataset to {save_path}")

def initialize_vram_monitor():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    return handle

def get_vram_usage(handle):
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    used = info.used / (1024 ** 3)  # Đổi sang GB
    total = info.total / (1024 ** 3)  # Đổi sang GB
    return used, total

def benchmark_encryption_gpu(encryption_function, batch_size, num_round):
    p = cp.zeros((batch_size, 64), dtype=cp.uint8)
    k = cp.zeros((batch_size, 80), dtype=cp.uint8)

    cp.cuda.Stream.null.synchronize()
    start = time.time()
    c = encryption_function(p, k, num_round)
    cp.cuda.Stream.null.synchronize()
    end = time.time()

    execution_time = end - start
    speed = batch_size / execution_time

    mempool = cp.get_default_memory_pool()
    memory_used = mempool.used_bytes() / (1024 ** 3)  # Đổi sang GB
    memory_peak = mempool.total_bytes() / (1024 ** 3)  # Đổi sang GB

    print(f"{batch_size} blocks, {num_round} rounds:")
    print(f"Execution time: {execution_time:.4f} seconds")
    print(f"Speed: {speed:,.2f} blocks/sec")
    print(f"GPU Memory used: {memory_used:.2f} GB")
    print(f"GPU Peak memory: {memory_peak:.2f} GB")

# def benchmark_make_train_data_gpu(encryption_func_gpu, plain_bits, key_bits, num_samples, nr, delta_state, delta_key, batch_size):
#     tracemalloc.start()
#     vram_handle = initialize_vram_monitor()
#     total_samples = 0
#     cp.cuda.Stream.null.synchronize()
#     start = time.time()

#     X_list, Y_list = [], []

#     for i in range(0, num_samples, batch_size):
#         current_batch = min(batch_size, num_samples - total_samples)
#         C, Y = make_train_data_gpu(encryption_func_gpu, plain_bits, key_bits, current_batch, nr, delta_state, delta_key)
#         X_list.append(C)
#         Y_list.append(Y)
#         cp.get_default_memory_pool().free_all_blocks()
#         total_samples += current_batch

#     cp.cuda.Stream.null.synchronize()
#     end = time.time()

#     current_ram, peak_ram = tracemalloc.get_traced_memory()
#     tracemalloc.stop()
#     used_vram, total_vram = get_vram_usage(vram_handle)

#     execution_time = end - start
#     samples_per_second = num_samples / execution_time
#     print(f"[GPU/Cupy] Generated {num_samples} samples: {execution_time:.4f}s ({samples_per_second:,.2f} samples/second)")
#     print(f"RAM used: {current_ram / (1024 ** 2):.2f} MB | Peak RAM: {peak_ram / (1024 ** 2):.2f} MB")
#     print(f"VRAM used: {used_vram:.2f} GB / {total_vram:.2f} GB")

#     X = cp.vstack(X_list)
#     Y = cp.hstack(Y_list)
#     return X, Y
def benchmark_make_train_data_gpu(encryption_func_gpu, plain_bits, key_bits, num_samples, nr, delta_state, delta_key, batch_size):
    tracemalloc.start()
    vram_handle = initialize_vram_monitor()
    total_samples = 0
    cp.cuda.Stream.null.synchronize()
    start = time.time()
    C_list, Y_list = [], []

    for i in range(0, num_samples, batch_size):
        current_batch = min(batch_size, num_samples - total_samples)
        C, Y = make_train_data_gpu(encryption_func_gpu, plain_bits, key_bits, current_batch, nr, delta_state, delta_key)
        C_list.append(C)
        Y_list.append(Y)
        
        used_vram, total_vram = get_vram_usage(vram_handle)
        print(f"[Batch {i // batch_size + 1}] VRAM used: {used_vram:.2f} GB / {total_vram:.2f} GB")
        
        cp.get_default_memory_pool().free_all_blocks()
        total_samples += current_batch

    cp.cuda.Stream.null.synchronize()
    end = time.time()
    tracemalloc.stop()
    used_vram, total_vram = get_vram_usage(vram_handle)

    execution_time = end - start
    samples_per_second = num_samples / execution_time
    print(f"[GPU/Cupy] Generated {num_samples} samples: {execution_time:.4f}s ({samples_per_second:,.2f} samples/second)")

    print(f"VRAM used: {used_vram:.2f} GB / {total_vram:.2f} GB")

    C = cp.vstack(C_list)
    Y = cp.hstack(Y_list)
    print(C.shape)
    return C, Y
def generate_sequence_datasets_by_round(cipher_name, encryption_function, plain_bits, key_bits, rounds_range, num_samples, delta_state, delta_key, batch_size, save_dir="../../data/"):
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n🚀 Generating {cipher_name} sequence datasets for rounds {list(rounds_range)}")

    for r in rounds_range:
        print(f"\n📦 Generating {cipher_name} round {r} by GPU...")
        C, Y = benchmark_make_train_data_gpu(
            encryption_func_gpu=encryption_function,
            plain_bits=plain_bits,
            key_bits=key_bits,
            num_samples=num_samples,
            nr=r,
            delta_state=delta_state,
            delta_key=delta_key,
            batch_size=batch_size,
        )
        # save_path = os.path.join(save_dir, f"{cipher_name}_seq_round{r}_samples{num_samples}.npz")
        # save_sequence_dataset_npz(C, Y, cipher_name, save_path=save_path)
    print(f"\n🏁 All {cipher_name} datasets generated and saved.")

if __name__ == "__main__":
    from ciphers.gpu import present80_gpu
    cipher_name = "present80"
    encryption_function = present80_gpu.encrypt 
    plain_bits = present80_gpu.plain_bits
    key_bits = present80_gpu.key_bits
    num_samples = 20_000_000
    batch_size = 1_000_000 # batch_size để chia nhỏ
    rounds_range = [10]
    delta_p_hex = "0x90"
    delta_plain = integer_to_binary_array_gpu(int(delta_p_hex, 16), plain_bits).reshape(-1)
    delta_key = 0

    print(f"Benchmarking encryption for {cipher_name}...")
    # benchmark_encryption_gpu(encryption_function, num_samples, num_round=11)

    generate_sequence_datasets_by_round(
        cipher_name=cipher_name,
        encryption_function=encryption_function,
        plain_bits=plain_bits,
        key_bits=key_bits,
        rounds_range=rounds_range,
        num_samples=num_samples,
        delta_state=delta_plain,
        delta_key=delta_key,
        batch_size=batch_size,
    )
