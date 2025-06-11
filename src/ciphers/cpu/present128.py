# Cipher PRESENT-128
import numpy as np

# Cipher parameters
plain_bits = 64
key_bits = 128
word_size = 4

def WORD_SIZE():
    return 64

# PRESENT S-Box and P-Box
Sbox = np.array([0xc, 0x5, 0x6, 0xb, 0x9, 0x0, 0xa, 0xd, 0x3, 0xe, 0xf, 0x8, 0x4, 0x7, 0x1, 0x2], dtype=np.uint8)

PBox = np.array([
    0, 16, 32, 48, 1, 17, 33, 49, 2, 18, 34, 50, 3, 19, 35, 51,
    4, 20, 36, 52, 5, 21, 37, 53, 6, 22, 38, 54, 7, 23, 39, 55,
    8, 24, 40, 56, 9, 25, 41, 57, 10, 26, 42, 58, 11, 27, 43, 59,
    12, 28, 44, 60, 13, 29, 45, 61, 14, 30, 46, 62, 15, 31, 47, 63
], dtype=np.uint8)

# Substitution (S-Box) layer
def SB(arr):
    num_words = arr.shape[1] // 4
    S = arr.copy()
    for i in range(num_words):
        to_sub = 0
        for j in range(4):
            pos = 4 * i + j
            to_sub += 2**(3 - j) * arr[:, pos]
        S[:, 4 * i:4 * (i + 1)] = np.unpackbits(Sbox[to_sub[:, None]], axis=1)[:, -4:]
    return S

# Permutation (P-Box) layer
def P(arr):
    permuted = arr[:, PBox]
    return permuted

# Key expansion
def expand_key(k, t):
    ks = [0] * t
    key = k.copy()
    for r in range(t):
        ks[r] = key[:, :64]
        # Rotate key left by 61 bits
        key = np.roll(key, -61, axis=1)
        # Apply S-Box to MSB 4 bits
        key[:, :4] = SB(key[:, :4])
        # XOR round counter into bits 62 to 66
        round_counter = np.unpackbits(np.uint8(r + 1))[3:]  # 5-bit round counter
        key[:, 59:64] ^= round_counter
    return ks

# Encryption function
def encrypt(p, k, r):
    ks = expand_key(k, r)
    c = p.copy()
    for i in range(r):
        c ^= ks[i]
        c = SB(c)
        c = P(c)
    return c

# Conversion utilities
def convert_to_binary(arr):
    X = np.zeros((len(arr) * WORD_SIZE(), len(arr[0])), dtype=np.uint8)
    for i in range(len(arr) * WORD_SIZE()):
        index = i // WORD_SIZE()
        offset = WORD_SIZE() - (i % WORD_SIZE()) - 1
        X[i] = (arr[index] >> offset) & 1
    X = X.transpose()
    return X

def convert_from_binary(arr, _dtype=np.uint64):
    num_words = arr.shape[1] // WORD_SIZE()
    X = np.zeros((len(arr), num_words), dtype=_dtype)
    for i in range(num_words):
        for j in range(WORD_SIZE()):
            pos = WORD_SIZE() * i + j
            X[:, i] += 2**(WORD_SIZE() - 1 - j) * arr[:, pos]
    return X

# Test vector check
def check_testvector():
    p = np.zeros((1, 64), dtype=np.uint8)
    k = np.zeros((1, 128), dtype=np.uint8)
    C = convert_from_binary(encrypt(p, k, 31))  # 31 rounds for PRESENT
    Chex = hex(C[0][0])
    assert Chex == "0x5565a5f9c4e1d2b4", f"Test vector failed: {Chex}"

# Execute test
check_testvector()
print("Test vector passed.")
