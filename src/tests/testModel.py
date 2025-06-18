# from ciphers.gpu import present80_gpu
from generator import cpu
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



def train_with_cpu():
    from ciphers.cpu import present80

    encryption_function = present80.encrypt
    plain_bits = present80.plain_bits
    key_bits = present80.key_bits
    num_samples = 10_000_000
    num_samples_val = 1_000_000

    nr=8
    delta = cpu.integer_to_binary_array_cpu(0x90000000000, plain_bits)
    delta_key = 0
    delta_plain = delta[:plain_bits]

    print("Training by CPU")
    C_cpu, Y_cpu = cpu.benchmark_make_train_data_cpu(encryption_function, num_samples, nr, plain_bits, key_bits, delta_plain, delta_key)
    C_cpu_val, Y_cpu_val = cpu.benchmark_make_train_data_cpu(encryption_function, num_samples_val, nr, plain_bits, key_bits, delta_plain, delta_key)


# get model architecture
    input_size = plain_bits
    model = dbitnet.make_model(2 * input_size)
    optimizer = tf.keras.optimizers.Adam(amsgrad=True)
    model.compile(optimizer=optimizer, loss='mse', metrics=['acc'])
    model_name= 'dbitnet'
    val_acc = tn.train_one_round(
        model,
        C_cpu, Y_cpu, C_cpu_val, Y_cpu_val,
        round_number = nr,
        epochs=EPOCHS,
        log_prefix="./",
        model_name=model_name,
        LR_scheduler=1e-5,
    )
    print(f'{model_name}, round {nr}. Best validation accuracy: {val_acc}', flush=True)

if __name__ == "__main__":
    train_with_cpu()