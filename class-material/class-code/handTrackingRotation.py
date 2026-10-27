import cv2
import mediapipe as mp
import numpy as np
import time
import math

# --- Camera ---
cap = cv2.VideoCapture(0)

# --- MediaPipe Hands ---
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6,
)

# --- Tuning (lower sensitivity & jitter) ---
ema_alpha = 0.22         # more responsive smoothing
angle_gain = 0.8         # higher = more sensitivity
deadband_deg = 1.0       # smaller deadband
max_deg_per_frame = 12.0 # allow quicker turns per frame

deadband = np.radians(deadband_deg)
max_step = np.radians(max_deg_per_frame)

def unwrap_angle(prev, new):
    """Make 'new' continuous vs 'prev' by wrapping diff to [-pi, pi]."""
    d = (new - prev + np.pi) % (2*np.pi) - np.pi
    return prev + d

def apply_deadband(delta, band):
    return 0.0 if abs(delta) < band else delta

def limit_step(prev, target, max_step):
    d = target - prev
    if d >  max_step: return prev + max_step
    if d < -max_step: return prev - max_step
    return target

# --- Smoothing state ---
yaw_ema = 0.0
pitch_ema = 0.0
roll_ema = 0.0

# --- Simple cube geometry (edge length ~2 units centered at origin) ---
cube_pts = np.array([
    [-1, -1, -1],
    [ 1, -1, -1],
    [ 1,  1, -1],
    [-1,  1, -1],
    [-1, -1,  1],
    [ 1, -1,  1],
    [ 1,  1,  1],
    [-1,  1,  1],
], dtype=np.float32)

cube_edges = [(0,1),(1,2),(2,3),(3,0), (4,5),(5,6),(6,7),(7,4), (0,4),(1,5),(2,6),(3,7)]

# --- Projection params ---
focal = 700.0       # camera focal length (pixels)
z_cam = 6.0         # distance of cube from camera (in "cube units")
cube_scale = 180.0  # overall size on screen (pixels)

pTime = 0

def rot_x(a):
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[1,0,0],
                     [0,ca,-sa],
                     [0,sa, ca]], dtype=np.float32)

def rot_y(a):
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[ ca,0, sa],
                     [  0,1,  0],
                     [-sa,0, ca]], dtype=np.float32)

def rot_z(a):
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[ca,-sa,0],
                     [sa, ca,0],
                     [ 0,  0,1]], dtype=np.float32)

def project_points(pts3d, cx, cy):
    """
    Perspective project 3D points to image plane centered at (cx, cy).
    Expects pts3d in camera coordinates where +x right, +y down, +z forward.
    """
    zs = pts3d[:, 2] + 1e-6
    us = (focal * (pts3d[:, 0] / zs)) + cx
    vs = (focal * (pts3d[:, 1] / zs)) + cy
    return np.stack([us, vs], axis=1).astype(np.int32)

while True:
    success, img = cap.read()
    if not success:
        break
    img = cv2.flip(img, 1)
    h, w, _ = img.shape
    cx_img, cy_img = w // 2, h // 2

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    # Defaults if no right hand is found
    yaw, pitch, roll = yaw_ema, pitch_ema, roll_ema

    if results.multi_hand_landmarks and results.multi_handedness:
        # multi_hand_landmarks and multi_handedness are aligned by index
        for handLms, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label  # 'Left' or 'Right'
            if label != 'Right':
                continue  # only the right hand controls the cube

            # --- Get normalized 3D landmark coords (x, y, z) ---
            pts = np.array([[lm.x, lm.y, lm.z] for lm in handLms.landmark], dtype=np.float32)

            # Landmarks for orientation
            wrist = pts[0]
            index_mcp = pts[5]
            middle_mcp = pts[9]
            pinky_mcp = pts[17]

            # Vectors in hand space
            forward = middle_mcp - wrist      # roughly points toward fingers
            right_v = index_mcp - pinky_mcp   # across the palm

            # Normalize
            def nrm(v):
                n = np.linalg.norm(v) + 1e-8
                return v / n

            f = nrm(forward)
            r = nrm(right_v)

            # MediaPipe: x right, y down, z negative away
            fx, fy, fz = f[0], f[1], f[2]  # fz negative when away

            # --- Raw angles from hand vectors ---
            yaw_raw   = math.atan2(fx, -fz)                              # Y (left/right)
            pitch_raw = math.atan2(-fy, math.sqrt(fx*fx + fz*fz))        # X (up/down)
            roll_raw  = math.atan2(r[1], r[0])                           # Z (tilt)

            # OPTIONAL: invert any axis if it feels backwards
            # yaw_raw = -yaw_raw
            # pitch_raw = -pitch_raw
            # roll_raw = -roll_raw

            # --- Sensitivity control (gain) ---
            yaw_raw   *= angle_gain
            pitch_raw *= angle_gain
            roll_raw  *= angle_gain

            # --- Unwrap vs previous filtered angles (avoid ±pi jumps) ---
            yaw_raw   = unwrap_angle(yaw_ema,   yaw_raw)
            pitch_raw = unwrap_angle(pitch_ema, pitch_raw)
            roll_raw  = unwrap_angle(roll_ema,  roll_raw)

            # --- Deadband (ignore micro jitter) ---
            yaw_target   = yaw_ema   + apply_deadband(yaw_raw   - yaw_ema,   deadband)
            pitch_target = pitch_ema + apply_deadband(pitch_raw - pitch_ema, deadband)
            roll_target  = roll_ema  + apply_deadband(roll_raw  - roll_ema,  deadband)

            # --- Slew-rate limit (cap per-frame change) ---
            yaw_target   = limit_step(yaw_ema,   yaw_target,   max_step)
            pitch_target = limit_step(pitch_ema, pitch_target, max_step)
            roll_target  = limit_step(roll_ema,  roll_target,  max_step)

            # --- Smooth toward target with EMA ---
            yaw_ema   = (1-ema_alpha) * yaw_ema   + ema_alpha * yaw_target
            pitch_ema = (1-ema_alpha) * pitch_ema + ema_alpha * pitch_target
            roll_ema  = (1-ema_alpha) * roll_ema  + ema_alpha * roll_target

            yaw, pitch, roll = yaw_ema, pitch_ema, roll_ema

            # Draw landmarks for visual feedback
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

            # only first detected right hand
            break

    # --- Build rotation from yaw(Y), pitch(X), roll(Z) ---
    R = rot_z(roll) @ rot_y(yaw) @ rot_x(pitch)

    # --- Transform cube to camera space ---
    cube_cam = (cube_pts * (cube_scale / 100.0)) @ R.T
    cube_cam[:, 2] += z_cam  # keep in front of camera

    # --- Project to 2D ---
    pts2d = project_points(cube_cam, cx_img, cy_img)

    # --- Draw cube edges ---
    for a, b in cube_edges:
        cv2.line(img, tuple(pts2d[a]), tuple(pts2d[b]), (50, 220, 255), 2)

    # --- FPS ---
    cTime = time.time()
    fps = 1.0 / max(cTime - pTime, 1e-6)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)

    # --- Show angles (deg) ---
    deg = lambda r: int(np.degrees(r))
    cv2.putText(img, f'Yaw: {deg(yaw)}',    (10, 70),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv2.putText(img, f'Pitch: {deg(pitch)}',(10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv2.putText(img, f'Roll: {deg(roll)}',  (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    # --- Display ---
    cv2.imshow("Hand-Controlled 3D Cube (Right Hand)", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
