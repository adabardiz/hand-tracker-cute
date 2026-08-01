import cv2
import numpy as np

try:
    from mediapipe.python.solutions import hands as mp_hands
except ImportError:
    import mediapipe.solutions.hands as mp_hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,             
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    mask = np.zeros((h, w), dtype=np.uint8)

    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) >= 2:
        hand1 = results.multi_hand_landmarks[0]
        hand2 = results.multi_hand_landmarks[1]
        
        # 8 = Index tip, 4 = Thumb tip
        pt1 = (int(hand1.landmark[8].x * w), int(hand1.landmark[8].y * h))
        pt2 = (int(hand2.landmark[8].x * w), int(hand2.landmark[8].y * h))
        pt3 = (int(hand2.landmark[4].x * w), int(hand2.landmark[4].y * h))
        pt4 = (int(hand1.landmark[4].x * w), int(hand1.landmark[4].y * h))
        
        polygon_points = np.array([pt1, pt2, pt3, pt4], np.int32)
        cv2.fillPoly(mask, [polygon_points], 255)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    color_effect = cv2.applyColorMap(gray, cv2.COLORMAP_SPRING)
    
    mask_3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    final_frame = np.where(mask_3d == 255, color_effect, frame)

    cv2.imshow('mediapipe tracker', final_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)