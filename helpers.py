import struct

def serialize_twist(lx: float, ly: float, lz: float, ax: float, ay: float, az: float) -> bytes:
    # CDR encapsulation header + geometry_msgs/Twist (2 x Vector3, 6 float64 little-endian)
    header = b"\x00\x01\x00\x00"
    payload = struct.pack("<dddddd", lx, ly, lz, ax, ay, az)

    return header + payload
