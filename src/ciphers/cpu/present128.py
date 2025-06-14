<<<<<<< HEAD
# cipher PRESENT-128 (failed)
from __future__ import print_function

s_box = (0xC, 0x5, 0x6, 0xB, 0x9, 0x0, 0xA, 0xD, 0x3, 0xE, 0xF, 0x8, 0x4, 0x7, 0x1, 0x2)

inv_s_box = (0x5, 0xE, 0xF, 0x8, 0xC, 0x1, 0x2, 0xD, 0xB, 0x4, 0x6, 0x3, 0x0, 0x7, 0x9, 0xA)

p_layer_order = [0, 16, 32, 48, 1, 17, 33, 49, 2, 18, 34, 50, 3, 19, 35, 51, 4, 20, 36, 52, 5, 21, 37, 53, 6, 22, 38,
                 54, 7, 23, 39, 55, 8, 24, 40, 56, 9, 25, 41, 57, 10, 26, 42, 58, 11, 27, 43, 59, 12, 28, 44, 60, 13,
                 29, 45, 61, 14, 30, 46, 62, 15, 31, 47, 63]

block_size = 64

ROUND_LIMIT = 32


def round_function(state, key):
    new_state = state ^ key
    state_nibs = []
    for x in range(0, block_size, 4):
        nib = (new_state >> x) & 0xF
        sb_nib = s_box[nib]
        state_nibs.append(sb_nib)
    # print(state_nibs)

    state_bits = []
    for y in state_nibs:
        nib_bits = [1 if t == '1'else 0 for t in format(y, '04b')[::-1]]
        state_bits += nib_bits
    # print(state_bits)
    # print(len(state_bits))

    state_p_layer = [0 for _ in range(64)]
    for p_index, std_bits in enumerate(state_bits):
        state_p_layer[p_layer_order[p_index]] = std_bits

    # print(len(state_p_layer), state_p_layer)

    round_output = 0
    for index, ind_bit in enumerate(state_p_layer):
        round_output += (ind_bit << index)

    # print(format(round_output, '#016X'))

    # print('')
    return round_output



def key_function_128(key, round_count):
    # print('Start: ', hex(key))
    # print('')

    r = [1 if t == '1'else 0 for t in format(key, '0128b')[::-1]]

    # print('k bits:', r)
    # print('')

    h = r[-61:] + r[:-61]

    # print('s bits:', h)
    # print('')

    round_key_int = 0
    # print('init round int:', hex(round_key_int))
    for index, ind_bit in enumerate(h):
        round_key_int += (ind_bit << index)
        # print('round:',index, '-', hex(round_key_int))

    # print('round_key_int', hex(round_key_int))
    # print('')

    upper_nibble = (round_key_int >> 124) & 0xF
    second_nibble = (round_key_int >> 120) & 0xF
    # print('upper_nibble:', upper_nibble)

    upper_nibble = s_box[upper_nibble]
    second_nibble = s_box[second_nibble]

    # print('upper_nibble sboxed', hex(upper_nibble))

    xor_portion = ((round_key_int >> 62) & 0x1F) ^ round_count
    # print('Count:', round_count)
    # print('XOR Value:', xor_portion)

    # print('Before:', hex(round_key_int))
    round_key_int = (round_key_int & 0x00FFFFFFFFFFFFF83FFFFFFFFFFFFFFF) + (upper_nibble << 124) + (second_nibble << 120) + (xor_portion << 62)
    # print('After: ', hex(round_key_int))

    return round_key_int


test_vectors_128 = {1:(0x00000000000000000000000000000000, 0x0000000000000000, 0x96db702a2e6900af),
                2:(0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF, 0x0000000000000000, 0x13238c710272a5d8),
                3:(0x00000000000000000000000000000000, 0xFFFFFFFFFFFFFFFF, 0x3c6019e5e5edd563),
                4:(0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x628d9fbd4218e5b4)}

for test_case in test_vectors_128:

    key_schedule = []
    current_round_key = test_vectors_128[test_case][0]
    round_state = test_vectors_128[test_case][1]

    # Key schedule
    for rnd_cnt in range(ROUND_LIMIT):
        # print(format(round_key, '020X'))
        # print(format(round_key >> 16, '016X'))
        key_schedule.append(current_round_key >> 64)
        current_round_key = key_function_128(current_round_key, rnd_cnt + 1)

    for rnd in range(ROUND_LIMIT - 1):
        # print('Round:', rnd)
        # print('State:', format(round_state, '016X'))
        # print('R_Key:', format(key_schedule[rnd], '016X'))
        round_state = round_function(round_state, key_schedule[rnd])

    round_state ^= key_schedule[31]

    if round_state == test_vectors_128[test_case][2]:
        print('Success', hex(round_state))
    else:
        print('Failure', hex(round_state))
=======
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
>>>>>>> main
