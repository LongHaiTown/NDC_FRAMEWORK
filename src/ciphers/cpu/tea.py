# Cipher TEA
import numpy as np

plain_bits = 64
key_bits = 128
word_size = 32

def add_mod(v1, v2, mod = 2**32):
    return (v1+v2)%mod

def encrypt(p, k, r):
    p = convert_from_binary(p)
    k = convert_from_binary(k)
    v0, v1 = p[:, 0], p[:, 1]
    k0, k1, k2, k3 = k[:, 0], k[:, 1], k[:, 2], k[:, 3]
    delta = 0x9E3779B9
    s = 0
    for i in range(r):
        s = add_mod(s, delta)
        v0 = add_mod(v0, add_mod((v1<<4), k0) ^ add_mod(v1, s) ^ add_mod((v1>>5), k1))
        v1 = add_mod(v1, add_mod((v0<<4), k2) ^ add_mod(v0, s) ^ add_mod((v0>>5), k3))
    return convert_to_binary([v0, v1])

def convert_to_binary(arr):
    X = np.zeros((len(arr) * 32, len(arr[0])), dtype = np.uint8)
    for i in range(len(arr) * 32):
        index = i // 32
        offset = 32 - (i % 32) - 1
        X[i] = (arr[index] >> offset) & 1
    X = X.transpose()
    return X

def convert_from_binary(arr, _dtype = np.uint32):
    num_words = arr.shape[1]//32
    X = np.zeros((len(arr), num_words), dtype = _dtype)
    for i in range(num_words):
        for j in range(32):
            pos = 32 * i + j
            X[:, i] += 2**(32-1-j) * arr[:, pos]
    return X
