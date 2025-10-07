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
    

#------------------------------------------------------------------------------#
def skin_mask(frame_bgr):
    '''Create mask of detected skin pixels'''
    # start by denoising with a blur
    sm = cv2.GaussianBlur(frame_bgr, (5, 5), 0)

    # convert to HSV and YCrCb (two color spaces where skin forms clusters)
    hsv = cv2.cvtColor(sm, cv2.COLOR_BGR2HSV)
    ycc = cv2.cvtColor(sm, cv2.COLOR_BGR2YCrCb)

    # broad HSV skin range 
    mask_hsv = cv2.inRange(hsv,
                           np.array([0, 30, 60], np.uint8),
                           np.array([25, 200, 255], np.uint8))
    # classic YCrCb skin range
    # Y — luminance, Cb — Chrominance-blue, Cr — Chrominance-red
    mask_ycc = cv2.inRange(ycc,
                           np.array([0, 133, 77], np.uint8),
                           np.array([255, 173, 127], np.uint8))
    # combine to only keep pixels that pass both thresholds
    mask = cv2.bitwise_and(mask_hsv, mask_ycc)

    # morphological cleanup (remove tiny dots, fill tiny holes)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)

    # feather edges for smoother blending
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    return mask  # uint8 (0..255)


    # lower = np.array([0, 48, 80], dtype = "uint8")
    # upper = np.array([20, 255, 255], dtype = "uint8")

#------------------------------------------------------------------------------#
def apply_smurf_tint(frame_bgr, mask, blue_bgr=(136, 204, 255), strength=0.65):
    '''Tint areas where mask>0 blue'''
    # default tint is smurf blue at 0.65 intensity
    tint = np.full_like(frame_bgr, blue_bgr, np.uint8)
    alpha = (mask.astype(np.float32) / 255.0) * strength
    alpha = alpha[..., None]
    out = (alpha * tint.astype(np.float32) + (1.0 - alpha) 
           * frame_bgr.astype(np.float32))
    
    return np.clip(out, 0, 255).astype(np.uint8)

#------------------------------------------------------------------------------#
# def load_cascades():
#     '''Load cv2 Haar cascades that detect facial features'''
#     # https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html

#------------------------------------------------------------------------------#
# https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker

#------------------------------------------------------------------------------#
def detect_faces_and_eyes() -> list[int]:
    '''Find faces and up to two eyes per face in a grayscale image'''
    # return a list of coordinates, vertices of rectangles for face/eye1/eye2

#------------------------------------------------------------------------------#
def estimate_head_tilt_degrees(eyes) -> int:
    '''Find head tilt if we have two eyes. If 2 eyes not found, return 0.'''
    # used to tilt the smurf hat with subject's head
    return 0

#------------------------------------------------------------------------------#
def main():
    print()