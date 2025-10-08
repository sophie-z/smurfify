'''
AUTHOR: Sophie Zeng
DATE: October 2025
PURPOSE: Smurf transmogrifier program ("Smurfify" camera) 
'''

import cv2
import numpy as np
import mediapipe as mp
import os

def overlay_rgba_on_bgr(frame, rgba_image, x, y, angle_deg=0, scale=1.0):
    '''Overlay a RBGA image into the webcam.'''
    # resize hat to requested scale (default 1.0)
    h, w = rgba_image.shape[:2]
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    overlay = cv2.resize(rgba_image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # rotate around the overlay center (if needed)
    if abs(angle_deg) > 0.1:
        M = cv2.getRotationMatrix2D((new_w // 2, new_h // 2), angle_deg, 1.0)
        overlay = cv2.warpAffine(overlay, M, (new_w, new_h), 
                                flags=cv2.INTER_LINEAR, 
                                borderMode=cv2.BORDER_TRANSPARENT)

    # clip image to screen so we don't draw out of bounds
    H, W = frame.shape[:2]
    x0 = max(0, x); y0 = max(0, y)
    x1 = min(W, x0 + new_w); y1 = min(H, y0 + new_h)
    if x0 >= x1 or y0 >= y1:
        return frame
    
    # crop both overlay and frame to the same size
    rx0 = x0 - x; ry0 = y0 - y
    rx1 = rx0 + (x1 - x0); ry1 = ry0 + (y1 - y0)
    roi_bgr = frame[y0:y1, x0:x1]
    patch = overlay[ry0:ry1, rx0:rx1]

    # if overlay has alpha channel, do alpha blending 
    # (mix hat colors with underlying frame accd to transparency level)
    if patch.shape[2] == 4:
        b, g, r, a = cv2.split(patch)
        alpha = (a.astype(np.float32) / 255.0)[..., None]  # shape (h, w, 1)
        rgb = cv2.merge([b, g, r]).astype(np.float32)
        base = roi_bgr.astype(np.float32)
        out = alpha * rgb + (1.0 - alpha) * base
        frame[y0:y1, x0:x1] = np.clip(out, 0, 255).astype(np.uint8)
    else:
        # if hat opaque, then we just paste the hat (uglier edges tho)
        frame[y0:y1, x0:x1] = patch[..., :3]
    return frame
        
#------------------------------------------------------------------------------#

# October 8th notes: instead of masking by color, try this:
# use mediapipe face detection to get face box, use edge detection to find skin 
# in face, use that to find skin color, then use skin color to find the rest 
# of the skin on the body 
# can use canny for edge detection in the bbox?

def skin_mask(frame_bgr):
    '''Create mask of pixels with detected skin color.'''
    img = cv2.GaussianBlur(frame_bgr, (5,5), 0)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    mask_hsv = cv2.inRange(hsv, np.array([0, 40, 70], np.uint8),
                                np.array([25,220,255], np.uint8))
    mask_ycc = cv2.inRange(ycc, np.array([0,135, 80], np.uint8),
                                np.array([255,170,130], np.uint8))

    mask = cv2.bitwise_and(mask_hsv, mask_ycc)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.GaussianBlur(mask, (5,5), 0)
    return mask

#------------------------------------------------------------------------------#
def colorize_skin_to_blue(frame_bgr, mask, hue_h=120, sat=200):
    '''Add blue hue to skin pixels.'''
    # hue_h: OpenCV hue is 0-179; blue ≈ 110-130 (120 is a solid blue)
    # sat: saturation is 0-255 (higher = more vivid)

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
LEFT_EYE_OUTER  = 263  # viewer's right eye outer corner (person's left)
RIGHT_EYE_OUTER = 33   # viewer's left eye outer corner (person's right)
FOREHEAD_TOP    = 10   # near top of forehead
# -----------------------------------------------------------------------------#
def lm_xy(landmark, width: int, height: int) -> tuple[int, int]:
    '''Convert MediaPipe normalized landmark to pixel coordinates.'''
    # mediapipe returns landmark coordinates normalized 0-1 from img size
    return int(landmark.x * width), int(landmark.y * height)

# -----------------------------------------------------------------------------#
def head_tilt_deg(landmarks, width: int, height: int) -> float:
    '''
    Estimate head tilt using the line through the two outer eye corners.
    Returns angle in degrees (positive = counter clock wise).
    '''
    try:
        x1, y1 = lm_xy(landmarks[RIGHT_EYE_OUTER], width, height)
        x2, y2 = lm_xy(landmarks[LEFT_EYE_OUTER],  width, height)
        dy, dx = (y2 - y1), (x2 - x1)
        # dy is neg to convert image coords to cartesian
        # y increases downward in our images, but cv2 cartesian thinks positive angles are CCW (y up)
        return float(np.degrees(np.arctan2(-dy, dx))) 
    
    except Exception:
        return 0.0 # if it can't find both eyes 
# -----------------------------------------------------------------------------#
def hat_anchor_and_width(landmarks, width: int, height: int,
                         face_bbox: tuple[int, int, int, int]
                         ) -> tuple[int, int, int]:
    '''
    Choose where to place the hat and how wide it should be.
    Horizontal center = use face bbox center (robust)
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
def gray_world_wb(bgr):
    '''
    '''
    b, g, r = cv2.split(bgr.astype(np.float32))
    mB, mG, mR = b.mean(), g.mean(), r.mean()
    gray = (mB + mG + mR) / 3.0 + 1e-6
    b *= gray / mB; g *= gray / mG; r *= gray / mR
    
    return np.clip(cv2.merge([b,g,r]), 0, 255).astype(np.uint8)

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
        raise RuntimeError("white_hat.png not found or missing alpha channel. " \
                          "Use a transparent PNG (RGBA).")

    # initialize MediaPipe (mesh for landmarks, detection for bboxes)
    mp_face_mesh = mp.solutions.face_mesh
    mp_face_det  = mp.solutions.face_detection

    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,      # process a video stream
        max_num_faces=5,              # support multiple people
        refine_landmarks=False,       # basic mesh is enough for hats
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    face_det = mp_face_det.FaceDetection(model_selection=0, min_detection_confidence=0.5)
    save_idx = 0

    while True:
        SELFIE_VIEW = False 

        ok, frame = cap.read()
        if not ok:
            break

        # flip so no mirror view
        frame = cv2.flip(frame, 1)

        H, W = frame.shape[:2]

        # build global skin mask (face + hands/arms) and apply blue tint
        frame = gray_world_wb(frame)
        mask = skin_mask(frame)
        # smurf = apply_smurf_tint(frame, mask, blue_bgr=(136, 204, 255), strength=0.65)
        smurf = colorize_skin_to_blue(frame, mask, hue_h=120, sat=120)

        # prepare RGB copy for MediaPipe (it expects RGB, not BGR)
        rgb = cv2.cvtColor(smurf, cv2.COLOR_BGR2RGB)

        # get face bounding boxes and landmarks 
        det_res  = face_det.process(rgb)
        mesh_res = face_mesh.process(rgb)

        # convert relative bboxes to pixel rectangles
        bboxes = []
        if det_res.detections:
            for det in det_res.detections:
                rb = det.location_data.relative_bounding_box
                fx = int(rb.xmin  * W)
                fy = int(rb.ymin  * H)
                fw = int(rb.width * W)
                fh = int(rb.height* H)
                bboxes.append((fx, fy, fw, fh))

        # for each face with landmarks, compute hat placement + rotation
        if mesh_res.multi_face_landmarks:
            for i, face_landmarks in enumerate(mesh_res.multi_face_landmarks):
                # pick a bbox for this face by index 
                bbox = bboxes[i] if i < len(bboxes) else (W//4, H//4, W//4, H//4)
                # estimate head tilt from eye outer corners
                angle = head_tilt_deg(face_landmarks.landmark, W, H)
                # compute hat anchor and size
                cx, top_y, hat_w = hat_anchor_and_width(face_landmarks.landmark, W, H, bbox)
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
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

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






# -----------------------------------------------------------------------------#
# def apply_smurf_tint(frame_bgr, mask, blue_bgr=(136, 204, 255), strength=0.65):
#     '''Tint pixels blue when mask>0'''
#     # default tint is smurf blue at 0.65 intensity
#     tint = np.full_like(frame_bgr, blue_bgr, np.uint8)
#     alpha = (mask.astype(np.float32) / 255.0) * strength
#     alpha = alpha[..., None]
#     out = (alpha * tint.astype(np.float32) + (1.0 - alpha) 
#            * frame_bgr.astype(np.float32))
    
#     # keep 0-255 and convert back to uint8
#     return np.clip(out, 0, 255).astype(np.uint8) 