# from ciphers.gpu import present80_gpu
from ciphers.cpu import present80 as present80_cpu
from generator import cipherpair_generator_cpu as cc
from generator import cipherpair_generator_gpu as cp

import cupy as cpy
import numpy as np

import tensorflow as tf
from models import dbitnet
from models import train_nets as tn
import logging
import pandas as pd
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler

ABORT_TRAINING_BELOW_ACC = 0.505   # if the validation accuracy reaches or falls below this limit, abort further training.
EPOCHS = 10                        # train for 10 epochs
NUM_SAMPLES = 10**7                 # create 10 million training samples
NUM_VAL_SAMPLES = 10**6             # create 1 million validation samples
BATCHSIZE = 5000   

def cupy_to_numpy_chunked(cupy_array, chunk_size=1_000_000):

    """
    Transfer a CuPy array to NumPy in chunks to avoid out-of-memory errors.
    
    Args:
        cupy_array: CuPy array to transfer.
        chunk_size: Number of elements per chunk (default: 1,000,000).
    
    Returns:
        NumPy array containing the transferred data.
    """
    total_size = cupy_array.size
    if total_size == 0:
        return np.array([])
    
    # Reshape to 1D for consistent slicing
    cupy_array_flat = cupy_array.ravel()
    result = []
    
    for start in range(0, total_size, chunk_size):
        end = min(start + chunk_size, total_size)
        # Transfer chunk to NumPy
        chunk = cupy_array_flat[start:end].get()
        result.append(chunk)
        # Synchronize to ensure GPU memory is freed
        cpy.cuda.Stream.null.synchronize()
    
    # Concatenate chunks and reshape to original shape
    numpy_array = np.concatenate(result)
    return numpy_array.reshape(cupy_array.shape)

def train_with_gpu():
    from ciphers.gpu import present80_gpu
    # from ciphers.cpu import present80_gpu

    encryption_function = present80_gpu.encrypt
    plain_bits = 64;
    key_bits =80
    num_samples = 1_000_000
    num_samples_val = 1_00_000

    nr=5

    def integer_to_binary_array(int_val, num_bits):
        return cp.array([int(i) for i in bin(int_val)[2:].zfill(num_bits)], dtype = cp.uint8).reshape(1, num_bits)

    delta = cp.integer_to_binary_array(0x90000000000, plain_bits)
    delta_key = 0
    delta_plain = delta[:plain_bits]

    print("hello world")

    data_generator = lambda num_samples, nr: cc.make_train_data(
        encryption_function, plain_bits, key_bits, num_samples, nr, delta_plain, delta_key
    )

    C_gpu, Y_gpu = cp.benchmark_make_train_data_gpu(encryption_function, num_samples, nr, plain_bits, key_bits)
    C_gpu_val, Y_gpu_val = cp.benchmark_make_train_data_gpu(encryption_function, num_samples_val, nr, plain_bits, key_bits)

# get model architecture
    input_size = plain_bits
    model = dbitnet.make_model(2 * input_size)
    optimizer = tf.keras.optimizers.Adam(amsgrad=True)
    model.compile(optimizer=optimizer, loss='mse', metrics=['acc'])
    model.summary()

    # Transfer data in chunks
    chunk_size = 1_000_000  # 1M elements per chunk
    X = cupy_to_numpy_chunked(C_gpu, chunk_size)
    del C_gpu  # Free GPU memory
    cpy.cuda.Stream.null.synchronize()
    Y = cupy_to_numpy_chunked(Y_gpu, chunk_size)
    del Y_gpu
    cpy.cuda.Stream.null.synchronize()
    X_val = cupy_to_numpy_chunked(C_gpu_val, chunk_size)
    del C_gpu_val
    cpy.cuda.Stream.null.synchronize()
    Y_val = cupy_to_numpy_chunked(Y_gpu_val, chunk_size)
    del Y_gpu_val
    cpy.cuda.Stream.null.synchronize()
    print(X.shape)
    # Clear TensorFlow session
    tf.keras.backend.clear_session()
    model_name= 'dbinet_gpu_dataset'
    val_acc = tn.train_one_round(
        model,
        X, Y, X_val, Y_val,
        round_number = nr,
        epochs=EPOCHS,
        log_prefix="./",
        model_name=model_name,
        LR_scheduler=1e-5,
    )
    print(f'{model_name}, round {nr}. Best validation accuracy: {val_acc}', flush=True)

def train_with_cpu():
    from ciphers.cpu import present80

    encryption_function = present80.encrypt
    plain_bits = 64;
    key_bits =80
    num_samples = 10_000_000
    num_samples_val = 1_000_000

    nr=8

    delta = cc.integer_to_binary_array(0x90000000000, plain_bits)
    delta_key = 0
    delta_plain = delta[:plain_bits]

    print("hello world")

    data_generator = lambda num_samples, nr: cc.make_train_data(
        encryption_function, plain_bits, key_bits, num_samples, nr, delta_plain, delta_key
    )

    C_gpu, Y_gpu = cc.benchmark_make_train_data_cpu(encryption_function, num_samples, nr, plain_bits, key_bits)
    C_gpu_val, Y_gpu_val = cc.benchmark_make_train_data_cpu(encryption_function, num_samples_val, nr, plain_bits, key_bits)


# get model architecture
    input_size = plain_bits
    model = dbitnet.make_model(2 * input_size)
    optimizer = tf.keras.optimizers.Adam(amsgrad=True)
    model.compile(optimizer=optimizer, loss='mse', metrics=['acc'])
    model.summary()

    model_name= 'dbinet_gpu_dataset'
    val_acc = tn.train_one_round(
        model,
        C_gpu, Y_gpu, C_gpu_val, Y_gpu_val,
        round_number = nr,
        epochs=EPOCHS,
        log_prefix="./",
        model_name=model_name,
        LR_scheduler=1e-5,
    )
    print(f'{model_name}, round {nr}. Best validation accuracy: {val_acc}', flush=True)

if __name__ == "__main__":
    train_with_cpu();