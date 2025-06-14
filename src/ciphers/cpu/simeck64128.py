# Cipher SIMECK-64-128
import numpy as np
from os import urandom

plain_bits = 64
key_bits = 128
word_size = 32

def WORD_SIZE():
    return(32)

MASK_VAL = 2**WORD_SIZE() - 1
def get_sequence(num_rounds):
    if num_rounds < 40:
        states = [1] * 5
    else:
        states = [1] * 6
    for i in range(num_rounds - 5):
        if num_rounds < 40:
            feedback = states[i + 2] ^ states[i]
        else:
            feedback = states[i + 1] ^ states[i]
        states.append(feedback)
    return tuple(states)

CONSTANT = 2**WORD_SIZE() - 4

def rol(x, k):
    return(((x << k) & MASK_VAL) | (x >> (WORD_SIZE() - k)))
def ror(x, k):
    return((x >> k) | ((x << (WORD_SIZE() - k)) & MASK_VAL))

def enc_one_round(p, k):
    tmp, c1 = p[0], p[1]
    tmp = tmp & rol(tmp,5)
    tmp = tmp ^ rol(p[0], 1)
    c1 = c1 ^ tmp
    c1 = c1 ^ k
    return(c1, p[0])

def dec_one_round(c, k):
    p0, p1 = c[0], c[1]
    tmp = tmp ^ rol(p1, GAMMA())
    p1 = tmp ^ c[0] ^ k
    p0 = c1
    return(p0, p1)



def encrypt(p, k, r):
    P = convert_from_binary(p)
    K = convert_from_binary(k).transpose()
    ks = expand_key(K, r)
    x, y = P[:, 0], P[:, 1];
    i=0
    for i in range(r):
        rk = ks[i]
        x,y = enc_one_round((x,y), rk);
    return convert_to_binary([x, y]);

def expand_key(k, t):
    sequence = get_sequence(t)
    ks = [0 for i in range(t)];
    #ks[0] = k[len(k)-1];
    states = list(reversed(k[:len(k)]));
    for i in range(t):
        ks[i] = states[0]
        l, r = states[1], states[0]
        l, r = enc_one_round((l, r), CONSTANT^sequence[i]);

        states.append(l)
        states.pop(0)
        states[0] = r
    return(ks);

def convert_to_binary(arr):
  X = np.zeros((len(arr) * WORD_SIZE(),len(arr[0])),dtype=np.uint8);
  for i in range(len(arr) * WORD_SIZE()):
    index = i // WORD_SIZE();
    offset = WORD_SIZE() - (i % WORD_SIZE()) - 1;
    X[i] = (arr[index] >> offset) & 1;
  X = X.transpose();
  return(X);

def convert_from_binary(arr, _dtype=np.uint32):
  num_words = arr.shape[1]//WORD_SIZE()
  X = np.zeros((len(arr), num_words),dtype=_dtype);
  for i in range(num_words):
    for j in range(WORD_SIZE()):
        pos = WORD_SIZE()*i+j
        X[:, i] += 2**(WORD_SIZE()-1-j)*arr[:, pos]
  return(X);

def check_testvectors():
  p = np.uint32([0x656b696c, 0x20646e75]).reshape(-1, 1)
  k = np.uint32([0x1b1a1918,0x13121110,0x0b0a0908,0x03020100]).reshape(-1, 1)
  pb = convert_to_binary(p)
  kb = convert_to_binary(k)
  c = convert_from_binary(encrypt(pb, kb, 44))
  assert np.all(c[0] == [0x45ce6902, 0x5f7ab7ed])

check_testvectors()