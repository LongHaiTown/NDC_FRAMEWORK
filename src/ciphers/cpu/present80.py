# Cipher PRESENT-80

import numpy as np

plain_bits = 64
key_bits = 80
word_size = 4

def WORD_SIZE():
<<<<<<< HEAD
    return(64)
=======
    return(64);
>>>>>>> main

Sbox = np.uint8([0xc, 0x5, 0x6, 0xb, 0x9, 0x0, 0xa, 0xd, 0x3, 0xe, 0xf, 0x8, 0x4, 0x7, 0x1, 0x2])
PBox = np.uint8([0, 16, 32, 48, 1, 17, 33, 49, 2, 18, 34, 50, 3, 19, 35, 51,
        4, 20, 36, 52, 5, 21, 37, 53, 6, 22, 38, 54, 7, 23, 39, 55,
        8, 24, 40, 56, 9, 25, 41, 57, 10, 26, 42, 58, 11, 27, 43, 59,
        12, 28, 44, 60, 13, 29, 45, 61, 14, 30, 46, 62, 15, 31, 47, 63])


def SB(arr):
    num_words = arr.shape[1]//4
    S = arr.copy()
    for i in range(num_words):
        to_sub = 0
        for j in range(4):
            pos = 4*i+j
            to_sub += 2**(3-j)*arr[:, pos]
        S[:, 4*i:4*(i+1)] = np.unpackbits(Sbox[to_sub[:, None]], axis = 1)[:, -4:]
    return S

def P(arr):
    arr[:, PBox] = arr[:, np.arange(64)]
    return arr



def expand_key(k, t):
    ks = [0 for i in range(t)];
    key = k.copy()
    for r in range(t):
        ks[r] = key[:, :64]
        key = np.roll(key, 19, axis = 1)
        key[:, :4] = SB(key[:, :4])
        key[:, -23:-15] ^= np.unpackbits(np.uint8(r+1))
    return ks


def encrypt(p, k, r):
    ks = expand_key(k, r)
    c = p.copy()
    for i in range(r-1):
        c ^= ks[i]
        c = SB(c)
        c = P(c)
    return c^ks[-1]


def convert_to_binary(arr):
  X = np.zeros((len(arr) * WORD_SIZE(),len(arr[0])),dtype=np.uint8);
  for i in range(len(arr) * WORD_SIZE()):
    index = i // WORD_SIZE();
    offset = WORD_SIZE() - (i % WORD_SIZE()) - 1;
    X[i] = (arr[index] >> offset) & 1;
  X = X.transpose();
  return(X);


def convert_from_binary(arr, _dtype=np.uint64):
  num_words = arr.shape[1]//WORD_SIZE()
  X = np.zeros((len(arr), num_words),dtype=_dtype);
  for i in range(num_words):
    for j in range(WORD_SIZE()):
        pos = WORD_SIZE()*i+j
        X[:, i] += 2**(WORD_SIZE()-1-j)*arr[:, pos]
  return(X);

<<<<<<< HEAD

=======
>>>>>>> main
def check_testvector():
    p = np.zeros((1, 64), dtype = np.uint8)
    k = np.zeros((1, 80), dtype = np.uint8)
    C = convert_from_binary(encrypt(p,k,32))
    Chex = hex(C[0][0])
    expected = '0x5579c1387b228445'
<<<<<<< HEAD
    assert Chex == "0x5579c1387b228445"
    print(C)
=======
    print("Computed: " + Chex)
    print("Expected: " + expected)
    assert Chex == "0x5579c1387b228445"
>>>>>>> main

check_testvector()


