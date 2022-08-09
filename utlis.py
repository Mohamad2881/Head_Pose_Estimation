import scipy.io as sio
import os
from math import cos, sin
import cv2
import mediapipe as mp
import pandas as pd

def get_img_names(file_path):
    # input:    relative path to .txt file with file names
    # output:   list of relative path names
    lines = [file[:-4] for file in os.listdir(file_path) if file.endswith('.jpg')]
    return lines


def get_ypr_from_mat(mat_path):
    # load mat file
    mat = sio.loadmat(mat_path)

    # Get Pose Parameters 
    # [pitch (phi), yaw (gamma), roll(theta), tdx, tdy, tdz, scale_factor]
    pose_params = mat['Pose_Para'][0]
    pitch, yaw, roll = pose_params[0], pose_params[1], pose_params[2]
    
    return yaw, pitch, roll


def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size = 100):
    yaw = -yaw
    # Convert degree to radians
    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),2)

    return img


def normalize_df(lands_df, x1, y1, x2, y2):
    """
    Normalize a DataFrame by subtracting a certain point (x1, y1) from all points
    and divide all points by a the distance between (x1, y1) and (x2, y2)
    
    Inputs:
        lands_df -- Pandas DataFrame contains landmark points to be normalized 
        x1 -- Series contain the x coordinates of the first point
        y1 -- Series contain the y coordinates of the first point
        x2 -- Series contain the x coordinates of the second point
        y2 -- Series contain the y coordinates of the second point
        
    Outputs:
        A Normalized DataFrame 
    """
    df_copy = lands_df.copy()

    # get all xs and subtract from it x1 
    df_copy.iloc[:, ::2] = df_copy.iloc[:, ::2].sub(x1, axis=0)
    
    # get all ys and subtract from it y1
    df_copy.iloc[:, 1::2] = df_copy.iloc[:, 1::2].sub(y1, axis=0)
    
    # get the distance between two points
    d_x = x2 - x1
    d_y = y2 - y1
    dist = (d_x**2 + d_y**2)**0.5
    
    # divide all points bt this distance
    df_copy = df_copy.div(dist, axis=0)
    
    return df_copy


def find_face_landmarks(img, max_num_faces,  draw_landmarks=True):
    """
    find face landmarks in an image
    
    Inputs:
        img_RGB -- image to find all face Landmarks in it
        max_num_faces -- Maximum number of faces to detect and find its landmarks
        draw_landmarks -- if True, all 468 landmarks will be drawn on the image
    Outputs:
        - an image with all landmarks drawn on it
        - A Data Frame where each row corresponds to a face being detected and 468 cols corresponds to the landmarks
        
    """
    # Convert BGR to RGB
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils

    drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=1)

    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=max_num_faces) as face_mesh:
        
        # get face landmarks for each detected face
        results = face_mesh.process(img_RGB)

        faces = []
        # check if any face is detected
        if results.multi_face_landmarks:
            # loop through detected faces
            for face_landmarks in results.multi_face_landmarks:
                if draw_landmarks:
                    mp_drawing.draw_landmarks(img, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS, drawing_spec, drawing_spec)
                face = {}
                # loop over 468 landmarks
                for idx, lm in enumerate(face_landmarks.landmark):
                    face[f'x{idx+1}'] = lm.x
                    face[f'y{idx+1}'] = lm.y

                faces.append(face)

    return img, pd.DataFrame(faces)
