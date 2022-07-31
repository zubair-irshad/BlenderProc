from utils import *
from scipy.spatial.transform import Rotation as R
from PIL import Image, ImageDraw, ImageFont
from typing import List
from tqdm import tqdm
import xml.etree.ElementTree as ET
import numpy as np
import os
import cv2

def get_aabb_coords(aabb_code):
    """Return a list of homogeneous 3D coords of the bbox. """
    [x1, y1, z1, x2, y2, z2] = aabb_code
    bbox_coords = np.array([[x1, y1, z1],
                           [x1, y2, z1],
                           [x2, y2, z1],
                           [x2, y1, z1],
                           [x1, y1, z2],
                           [x1, y2, z2],
                           [x2, y2, z2],
                           [x2, y1, z2]])
    bbox_coords = np.concatenate((bbox_coords, np.ones((8, 1))), 1)
    return bbox_coords

def project(intrinsic_mat,  # [3x3]
            pose_mat,  # [4x4], world coord -> camera coord
            box_coords  # [8x4] (x,y,z,1)
            ):
    """ Project 3D homogeneous coords to 2D coords. 
        in_front=True if at least 4 coords are in front of the camera. """

    #-- From world space to camera space.  
    box_coords_cam = np.matmul(pose_mat, np.transpose(box_coords, [1, 0]))  # [4x8]
    for i in range(8):
        box_coords_cam[:3, i] /= box_coords_cam[3, i]
    
    #-- Handle points behind camera: negate x,y if z<0
    for i in range(8):
        if box_coords_cam[2, i] < 0:
            box_coords_cam[0, i] = -box_coords_cam[0, i]
            box_coords_cam[1, i] = -box_coords_cam[1, i]
    
    #-- From camera space to picture space. 
    box_coords_pic = np.matmul(intrinsic_mat, box_coords_cam[:3, :])  # [3x8]
    for i in range(8):
        box_coords_pic[:2, i] /= box_coords_pic[2, i]
    final_coords = np.array(np.transpose(box_coords_pic[:2, :], [1, 0]).astype(np.int16))

    #-- Report whether the whole object is in front of camera.
    in_front = True
    if np.sum(box_coords_cam[2,:]>0) < 4: # not in_front: less than 4 coords are in front of the camera
        in_front = False 

    return final_coords, in_front

def draw_bbox(img, pic_coords, color, label):
    """Draw bounding boxes and labels on the image. """
    if isinstance(img, np.ndarray):
        cv2.line(img, pic_coords[0], pic_coords[1], color, 1)
        cv2.line(img, pic_coords[1], pic_coords[2], color, 1)
        cv2.line(img, pic_coords[2], pic_coords[3], color, 1)
        cv2.line(img, pic_coords[3], pic_coords[0], color, 1)
        cv2.line(img, pic_coords[0], pic_coords[4], color, 1)
        cv2.line(img, pic_coords[1], pic_coords[5], color, 1)
        cv2.line(img, pic_coords[2], pic_coords[6], color, 1)
        cv2.line(img, pic_coords[3], pic_coords[7], color, 1)
        cv2.line(img, pic_coords[4], pic_coords[5], color, 1)
        cv2.line(img, pic_coords[5], pic_coords[6], color, 1)
        cv2.line(img, pic_coords[6], pic_coords[7], color, 1)
        cv2.line(img, pic_coords[7], pic_coords[4], color, 1)
        cv2.putText(img, label, (pic_coords[0][0], pic_coords[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    elif isinstance(img, Image):
        draw = ImageDraw.Draw(img)
        draw.line((*pic_coords[0], *pic_coords[1]), fill=color)
        draw.line((*pic_coords[1], *pic_coords[2]), fill=color)
        draw.line((*pic_coords[2], *pic_coords[3]), fill=color)
        draw.line((*pic_coords[3], *pic_coords[0]), fill=color)
        draw.line((*pic_coords[0], *pic_coords[4]), fill=color)
        draw.line((*pic_coords[1], *pic_coords[5]), fill=color)
        draw.line((*pic_coords[2], *pic_coords[6]), fill=color)
        draw.line((*pic_coords[3], *pic_coords[7]), fill=color)
        draw.line((*pic_coords[4], *pic_coords[5]), fill=color)
        draw.line((*pic_coords[5], *pic_coords[6]), fill=color)
        draw.line((*pic_coords[6], *pic_coords[7]), fill=color)
        draw.line((*pic_coords[7], *pic_coords[4]), fill=color)
        font = ImageFont.truetype(r'/usr/share/fonts/truetype/freefont/FreeMono.ttf', 20)
        draw.text(pic_coords[0], label, color, font)

def project_bbox_to_image(img, # PIL image
                          intrinsic_mat,  # [3x3]
                          pose,  # [4x4], world coord -> camera coord
                          aabb_codes,  # [Nx6]
                          labels # [n x str]
                          ):
    """Project a list of bounding boxes to an image. Return the image wiht bounding boxes drawn. """
    img_with_bbox = img.copy()
    for aabb_code, label in zip(aabb_codes, labels):
        bbox_coords = get_aabb_coords(aabb_code)
        pic_coords, in_front = project(intrinsic_mat, pose, bbox_coords)
        color = np.random.choice(range(256), size=3)
        if in_front:
            draw_bbox(img_with_bbox, pic_coords, (int(color[0]), int(color[1]), int(color[2])), label)
    return img_with_bbox
