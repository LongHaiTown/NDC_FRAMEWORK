# Cipher ASCON
from typing import Tuple, List

# S-box sử dụng trong Ascon
SBOX = [0x4, 0xb, 0x2, 0x7, 0x5, 0x0, 0xe, 0xa, 0xf, 0x1, 0x3, 0x9, 0x8, 0x6, 0xc, 0xd]

# Hàm áp dụng S-box
def sbox(state: List[int]) -> List[int]:
    return [SBOX[b] for b in state]

# Hàm hoán vị
def permute(state: List[int], rounds: int) -> List[int]:
    for round_idx in range(rounds):
        state = sbox(state)  # Áp dụng S-box
        state[0] ^= round_idx  # XOR với hằng số vòng
    return state

# Thuật toán Ascon AEAD
def ascon_encrypt(key: bytes, nonce: bytes, plaintext: bytes, associated_data: bytes) -> Tuple[bytes, bytes]:
    """
    Thuật toán mã hóa Ascon AEAD.
    - key: Khóa bí mật (128 bit)
    - nonce: Nonce (128 bit)
    - plaintext: Dữ liệu cần mã hóa
    - associated_data: Dữ liệu liên quan không mã hóa
    - Trả về: ciphertext và tag
    """
    # 1. Khởi tạo trạng thái
    state = [0] * 40  # Trạng thái 320-bit (40 byte)
    state[:16] = list(nonce[:16])  # Nonce
    state[16:32] = list(key[:16])  # Khóa
    state = permute(state, 12)  # Hoán vị khởi tạo

    # 2. Xử lý dữ liệu liên quan (associated data)
    if associated_data:
        for i in range(0, len(associated_data), 8):
            block = associated_data[i:i + 8]
            for j in range(len(block)):
                state[j] ^= block[j]
            state = permute(state, 6)
        # Padding dữ liệu liên quan
        if len(associated_data) % 8 != 0:
            state[len(associated_data) % 8] ^= 0x80
        state = permute(state, 6)

    # 3. Mã hóa plaintext
    ciphertext = bytearray()
    for i in range(0, len(plaintext), 8):
        block = plaintext[i:i + 8]
        temp = bytearray(state[:len(block)])
        for j in range(len(block)):
            state[j] ^= block[j]
            temp[j] ^= block[j]
        ciphertext.extend(temp)
        state = permute(state, 6)

    # Padding plaintext
    if len(plaintext) % 8 != 0:
        state[len(plaintext) % 8] ^= 0x80

    # 4. Tạo mã xác thực (tag)
    state[16:32] = list(key[:16])  # Đưa khóa vào lại trạng thái
    state = permute(state, 12)
    tag = bytes(state[:16])  # Tag 128-bit

    return bytes(ciphertext), tag


key = bytes.fromhex("000102030405060708090a0b0c0d0e0f")
nonce = bytes.fromhex("000102030405060708090a0b0c0d0e0f")
plaintext = bytes.fromhex("0001020304050607")
associated_data = bytes.fromhex("00010203")

ciphertext, tag = ascon_encrypt(key, nonce, plaintext, associated_data)
print("Ciphertext:", ciphertext.hex())
print("Tag:", tag.hex())
