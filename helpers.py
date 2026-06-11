import struct

def serialize_vector3(x: float, y: float, z: float) -> bytes:
    # CDR encapsulation header
    header = b"\x00\x01\x00\x00"

    # 3 float64 little-endian
    payload = struct.pack("<ddd", x, y, z)

    return header + payload