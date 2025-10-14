'''
AUTHOR: Sophie Zeng
DATE: October 2025
PURPOSE: Smurf transmogrifier program ("Smurfify" camera) 
'''

import cv2
import numpy as np
import mediapipe as mp
import os

def overlay_rgba_on_bgr(frame: np.ndarray, rgba_image: np.ndarray, x: int, 
                        y: int, angle_deg: float = 0, 
                        scale: float = 1.0) -> np.ndarray:
    '''Overlay an RBGA image into the webcam.'''
    # resize
    h, w = rgba_image.shape[:2]
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    overlay = cv2.resize(rgba_image, (new_w, new_h), 
                         interpolation=cv2.INTER_AREA)

    # rotate around the overlay center if needed
    if abs(angle_deg) > 0.1:
        M = cv2.getRotationMatrix2D((new_w // 2, new_h // 2), angle_deg, 1.0)
        # keep full canvas size; transparent border
        overlay = cv2.warpAffine(
            overlay, M, (new_w, new_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_TRANSPARENT
        )

    # clip image to screen so we don't draw out of bounds
    H, W = frame.shape[:2]
    x0 = max(0, x); y0 = max(0, y)
    x1 = min(W, x + new_w); y1 = min(H, y + new_h)
    if x0 >= x1 or y0 >= y1:
        return frame  

    # calculate matching crop in overlay space 
    rx0 = max(0, x0 - x)
    ry0 = max(0, y0 - y)

    # compute how many pixels we CAN take from overlay from (rx0,ry0)
    # if hat is rotated or off screen the overlay gets messed up
    # so we figure out visible overlap of the hat+frame and clip everything else 
    # prevents trying to blend hat alpha+frame’s ROI that are different sizes
    max_w = new_w - rx0
    max_h = new_h - ry0
    need_w = x1 - x0
    need_h = y1 - y0

    w_clip = min(max_w, need_w)
    h_clip = min(max_h, need_h)

    # adjust frame ROI to exactly w_clip x h_clip
    x1 = x0 + w_clip
    y1 = y0 + h_clip

    roi_bgr = frame[y0:y1, x0:x1]
    patch  = overlay[ry0:ry0 + h_clip, rx0:rx0 + w_clip]

    # if overlay has alpha channel, do alpha blending 
    # (mix hat colors with underlying frame accd to transparency level)
    if patch.shape[2] == 4:
        b, g, r, a = cv2.split(patch)
        alpha = (a.astype(np.float32) / 255.0)[..., None]
        rgb   = cv2.merge([b, g, r]).astype(np.float32)
        base  = roi_bgr.astype(np.float32)
        out   = alpha * rgb + (1.0 - alpha) * base
        frame[y0:y1, x0:x1] = np.clip(out, 0, 255).astype(np.uint8)
    else:
        frame[y0:y1, x0:x1] = patch[..., :3]
    return frame
        
#------------------------------------------------------------------------------#
def build_skin_mask_via_facecolor(frame_bgr: np.ndarray, 
                                  face_boxes: list[tuple[int, int, int, int]]
                                  ) -> np.ndarray:
    '''
    Estimate face skin HSV from face edges and turn into full skin mask.
    If no face found (or poor sample), return 0s (x/y/w/h)
    '''
    if not face_boxes:
        return np.zeros(frame_bgr.shape[:2], dtype=np.uint8)

    # pick largest bbox 
    areas = [w*h for (_,_,w,h) in face_boxes]
    face_box = face_boxes[int(np.argmax(areas))]

    hsv_center = median_skin_hsv_from_face(frame_bgr, face_box)
    if hsv_center is None:
        return np.zeros(frame_bgr.shape[:2], dtype=np.uint8)

    # specify tolerance on HSV values 
    # 6 for hue bc skin color shouldn't vary much; leaves eyes/lips mostly alone
    # 50 for saturation and value to adjust for lighting and stuff
    return global_skin_mask_from_color(frame_bgr, hsv_center, tol_h=6, 
                                       tol_s=50, tol_v=50)

#------------------------------------------------------------------------------#
def median_skin_hsv_from_face(frame_bgr: np.ndarray, 
                              face_box: tuple[int, int, int, int]
                              ) -> tuple[int, int, int] | None:
    '''
    Inside the face box, keep only smooth (non-edge) regions (likely skin)
    and compute the median HSV color over those pixels.
    Returns (h0, s0, v0) in ranges (H:0..179, S,V:0..255) or None.
    '''
    H, W = frame_bgr.shape[:2]
    fx, fy, fw, fh = face_box
    fx = max(0, fx); fy = max(0, fy)
    fw = max(1, min(fw, W - fx)); fh = max(1, min(fh, H - fy))

    face = frame_bgr[fy:fy+fh, fx:fx+fw].copy()
    if face.size == 0:
        return None

    # canny edge detection to find face in bbox
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(gray, 60, 150)
    # morph cleanup
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    edges = cv2.dilate(edges, k, iterations=1)
    smooth_mask = cv2.bitwise_not(edges) 

    hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    sat_gate = (s > 30)
    val_gate = (v > 60)
    m = (smooth_mask > 0) & sat_gate & val_gate

    # if sample not big enough, return none
    if np.count_nonzero(m) < 200:
        return None

    h0 = int(np.median(h[m]))
    s0 = int(np.median(s[m]))
    v0 = int(np.median(v[m]))
    return (h0, s0, v0)

#------------------------------------------------------------------------------#
def global_skin_mask_from_color(frame_bgr: np.ndarray, 
                                hsv_center: tuple[int, int, int], 
                                tol_h: int = 6, tol_s: int = 50, 
                                tol_v: int = 50) -> np.ndarray:
    '''
    Given center HSV color (from the face), find similar colors across frame.
    Uses circular distance for hue & absolute distance for S/V. 
    Returns uint8 mask.
    '''
    if hsv_center is None:
        return np.zeros(frame_bgr.shape[:2], dtype=np.uint8)

    h0, s0, v0 = hsv_center
    img = cv2.GaussianBlur(frame_bgr, (5,5), 0)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    Hc, Sc, Vc = cv2.split(hsv)

    dh = np.minimum(np.abs(Hc.astype(np.int16) - h0),
                    180 - np.abs(Hc.astype(np.int16) - h0))
    ds = np.abs(Sc.astype(np.int16) - s0)
    dv = np.abs(Vc.astype(np.int16) - v0)

    mask = (dh <= tol_h) & (ds <= tol_s) & (dv <= tol_v)
    mask = (mask.astype(np.uint8) * 255)

    # form skin mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.GaussianBlur(mask, (5,5), 0)
    return mask

#------------------------------------------------------------------------------#
def colorize_skin_to_blue(frame_bgr: np.ndarray, mask: np.ndarray, 
                          hue_h: int = 105, sat: int = 140) -> np.ndarray:
    '''Add blue hue to skin pixels.'''
    # hue_h: OpenCV hue is 0-179; 105 is around smurf blue
    # sat: saturation is 0-255; 140 is about the vividness of a smurf

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    m = (mask > 0)

    h[m] = np.uint8(np.clip(hue_h, 0, 179))
    s[m] = np.uint8(np.clip(sat,   0, 255))
    recol = cv2.merge([h, s, v])
    return cv2.cvtColor(recol, cv2.COLOR_HSV2BGR)

# -----------------------------------------------------------------------------#
# Face Mesh provides 468 landmarks. we’ll use some of em here:
# 1) right/left outer eye corners to estimate head tilt
# 2) top-of-forehead landmark for vertical hat anchor
# storage.googleapis.com
# /mediapipe-assets/documentation/mediapipe_face_landmark_fullsize.png
LEFT_EYE_OUTER  = 263  # viewer's right eye outer corner (person's left)
RIGHT_EYE_OUTER = 33   # viewer's left eye outer corner (person's right)
FOREHEAD_TOP    = 10   # near top of forehead

# -----------------------------------------------------------------------------#
def lm_xy(landmark: tuple[int, int], width: int, height: int) -> tuple[int, int]:
    '''Convert MediaPipe normalized landmark to usable pixel coordinates.'''
    # mediapipe returns landmark coordinates normalized 0-1 from img size
    return int(landmark.x * width), int(landmark.y * height)

# -----------------------------------------------------------------------------#
def head_tilt_deg(landmarks: list, width: int, height: int) -> float:
    '''
    Estimate head tilt using the line through the two outer eye corners.
    Returns angle in degrees (positive = counter clock wise).
    '''
    try:
        x1, y1 = lm_xy(landmarks[RIGHT_EYE_OUTER], width, height)
        x2, y2 = lm_xy(landmarks[LEFT_EYE_OUTER],  width, height)
        dy, dx = (y2 - y1), (x2 - x1)
        # dy is neg to convert image coords to cartesian
        # y increases going downward in our images
        # but cv2 thinks positive angles are CCW (y going upward), so we flip
        return float(np.degrees(np.arctan2(-dy, dx))) 
    
    except Exception:
        return 0.0 # if it can't find both eyes 
# -----------------------------------------------------------------------------#
def hat_anchor_and_width(landmarks: list, width: int, height: int,
                         face_bbox: tuple[int, int, int, int]
                         ) -> tuple[int, int, int]:
    '''
    Choose where to place the hat and how wide it should be.
    Horizontal center = use face bbox center
    Vertical anchor = above the FOREHEAD_TOP landmark (top of bbox if missing)
    Returns (anchor_center_x, anchor_top_y, hat_width_pixels).
    '''
    fx, fy, fw, fh = face_bbox # rectangle around face from mediapipe 
    cx = fx + fw // 2  # face center x
    try:
        _, top_y = lm_xy(landmarks[FOREHEAD_TOP], width, height) # find forehead
    except Exception:
        top_y = fy  # use top of facebox for bottom of hat if forehead not found
    
    hat_w = int(1.5 * fw)
    return cx, top_y, hat_w

# -----------------------------------------------------------------------------#
def main():
    # open macbook camera
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Could not open camera. Check permissions in " \
                              "System Settings > Privacy & Security > Camera.")

    # optional: set resolution for performance 
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

    # load hat image (must be RGBA with transparency)
    hat_rgba = cv2.imread('white_hat.png', cv2.IMREAD_UNCHANGED)
    if hat_rgba is None or (hat_rgba.shape[2] != 4):
        raise RuntimeError("white_hat.png not found or missing alpha channel." \
                          " Use a transparent PNG (RGBA).")

    # initialize mediapipe (mesh for landmarks, detection for bboxes)
    mp_face_mesh = mp.solutions.face_mesh
    mp_face_det  = mp.solutions.face_detection

    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,      # process a video stream
        max_num_faces=5,              # support multiple people
        refine_landmarks=False,       # basic mesh is enough for hats
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    face_det = mp_face_det.FaceDetection(model_selection=0, 
                                         min_detection_confidence=0.5)
    save_idx = 0

    while True:
        SELFIE_VIEW = False # unmirror webcam
        ok, frame = cap.read()
        if not ok:
            break

        # flip so no mirror view
        frame = cv2.flip(frame, 1)
        H, W = frame.shape[:2]

        # detect faces first
        rgb_for_mp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        det_res  = face_det.process(rgb_for_mp)
        mesh_res = face_mesh.process(rgb_for_mp)
        # build list of pixel-space bboxes
        bboxes = []
        if det_res.detections:
            for det in det_res.detections:
                rb = det.location_data.relative_bounding_box
                fx = int(rb.xmin  * W); fy = int(rb.ymin  * H)
                fw = int(rb.width * W); fh = int(rb.height* H)
                bboxes.append((fx, fy, fw, fh))
        # new skin mask
        mask = build_skin_mask_via_facecolor(frame, bboxes)
        # recolor skin 
        smurf = colorize_skin_to_blue(frame, mask, hue_h=105, sat=140)

        # for each face with landmarks, compute hat placement + rotation
        if mesh_res.multi_face_landmarks:
            for i, face_landmarks in enumerate(mesh_res.multi_face_landmarks):
                # pick a bbox for this face by index 
                bbox = bboxes[i] if i < len(bboxes) else (W//4, H//4, 
                                                          W//4, H//4)
                # estimate head tilt from eye outer corners
                angle = head_tilt_deg(face_landmarks.landmark, W, H)
                # compute hat anchor and size
                cx, top_y, hat_w = hat_anchor_and_width(face_landmarks.landmark, 
                                                        W, H, bbox)
                # scale hat to the computed width (keep aspect ratio)
                scale = hat_w / hat_rgba.shape[1]
                hat_h = int(hat_rgba.shape[0] * scale)
                # place hat so its bottom center sits a bit above forehead
                hat_x = int(cx - (hat_rgba.shape[1] * scale) / 2)
                hat_y = int(top_y - 0.75 * hat_h)  # tweak to raise/lower  hat
                # overlay hat! (rotated to match head tilt)
                smurf = overlay_rgba_on_bgr(smurf, hat_rgba, hat_x, hat_y,
                                            angle_deg=angle, scale=scale)

        # UI text and display
        cv2.putText(smurf, "Smurf Cam (MediaPipe)  |  q: quit  s: save",
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 
                    2, cv2.LINE_AA)

        cv2.imshow("Smurf Cam (MediaPipe)", smurf)
        key = cv2.waitKey(1) & 0xFF

        # q to quit and s to save frame
        if key == ord('q'):
            break
        elif key == ord('s'):
            out_name = f"smurf_{save_idx:03d}.png"
            cv2.imwrite(out_name, smurf)
            print("Saved", out_name)
            save_idx += 1

    # cleanup
    cap.release()
    cv2.destroyAllWindows()

main()