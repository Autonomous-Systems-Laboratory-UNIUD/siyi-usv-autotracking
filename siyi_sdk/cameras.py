"""
This script defines the camera specs
"""

RESOLUTIONS = {
    "4K": (3840, 2160),
    "2K": (2560, 1440),
    "1080p": (1920, 1080),
    "720p": (1280, 720),
    "480p": (640, 480) 
}

class A8MINI:
    """
    The following bit rates seem to apply for the various streams:
    - recording stream: min 15k kbps, max 30k kbps
    - main/sub stream: min 1001 kbps, max 4k kpbs
    --
    Streaming resolutions: 480p, 720p, 1080p, 2K
    """
    MAX_YAW_DEG = 135.0
    MIN_YAW_DEG = -135.0
    MAX_PITCH_DEG = 25.0
    MIN_PITCH_DEG = -90.0
    MAX_ZOOM = 6.0
    RECORDING_RESOLUTIONS = [
        RESOLUTIONS['4K'],
        RESOLUTIONS['2K'],
        RESOLUTIONS['1080p'],
        RESOLUTIONS['720p']
    ]

class ZR10:
    MAX_YAW_DEG = 135.0
    MIN_YAW_DEG = -135.0
    MAX_PITCH_DEG = 25.0
    MIN_PITCH_DEG = -90.0
    MAX_ZOOM = 30.0 # 10 optical * 3 digital
    RECORDING_RESOLUTIONS = [
        RESOLUTIONS['2K'],
        RESOLUTIONS['1080p'],
        RESOLUTIONS['720p']
    ]
