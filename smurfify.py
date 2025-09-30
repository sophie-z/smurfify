'''
AUTHOR: Sophie Zeng
DATE: October 2025
PURPOSE: Smurf transmogrifier program ("Smurfify" camera) 
'''

import cv2
import numpy as np
import os

def overlay_rgba_on_bgr(frame, rgba_image, x, y, angle_deg=0, scale=1.0):
    '''Overlay a RBGA image into the webcam.'''
    return 0

#------------------------------------------------------------------------------#
def skin_mask(frame_bgr):
    '''Create mask of detected skin pixels'''
    # define boundaries of the HSV pixel intensities to be considered skin
    # disclaimer: this may be very racist?
    lower = np.array([0, 48, 80], dtype = "uint8")
    upper = np.array([20, 255, 255], dtype = "uint8")


#------------------------------------------------------------------------------#
def apply_smurf_tint(frame_bgr, mask, blue_bgr=(136, 204, 255), strength=0.65):
    '''Tint previously detected skin pixels blue'''
    # default color is smurf blue at 0.65 intensity

#------------------------------------------------------------------------------#
def load_cascades():
    '''Load cv2 Haar cascades that detect facial features'''
    # https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html

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