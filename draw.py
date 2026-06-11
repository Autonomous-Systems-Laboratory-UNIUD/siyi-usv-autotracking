import cv2

def draw_tracking_overlay(frame, detection):
    display = frame.copy()

    if detection is not None:
        cv2.rectangle(
            display,
            (detection.x1, detection.y1),
            (detection.x2, detection.y2),
            (0, 0, 255),
            2,
        )
        label = f"{detection.label} {detection.confidence:.2f}"
        cv2.putText(
            display,
            label,
            (detection.x1, max(24, detection.y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    return display