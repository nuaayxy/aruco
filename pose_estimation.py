'''
Sample Usage:-
python pose_estimation.py --K_Matrix calibration_matrix.npy --D_Coeff distortion_coefficients.npy --type DICT_5X5_100
'''


import numpy as np
import cv2
import sys
from utils import ARUCO_DICT
import argparse
import time
import socket
from scipy.spatial.transform import Rotation as R

UDP_IP = "127.0.0.1"
UDP_PORT = 8080
MESSAGE = "Hello, World!"


sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # UDP


def pose_esitmation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients):

    '''
    frame - Frame from the video stream
    matrix_coefficients - Intrinsic matrix of the calibrated camera
    distortion_coefficients - Distortion coefficients associated with your camera

    return:-
    frame - The frame with the axis drawn on it
    '''

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters_create()


    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, cv2.aruco_dict,parameters=parameters)

    rotation_matrix = np.array([[0, 0, 0],
                                [0, 0, 0],
                                [0, 0, 0]],
                                dtype=float)
        # If markers are detected
    if len(corners) > 0:
        for i in range(0, len(ids)):
            # Estimate pose of each marker and return the values rvec and tvec---(different from those of camera coefficients)
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.15, matrix_coefficients,
                                                                       distortion_coefficients)
            # Draw a square around the markers
            cv2.aruco.drawDetectedMarkers(frame, corners, ids,  borderColor=(255, 0, 255)) 

            rotation_matrix[:3, :3], _ = cv2.Rodrigues(rvec)
            
            # convert the matrix to a quaternion
            quaternion  = (R.from_matrix(rotation_matrix)).as_quat()
            tvec  = tvec.flatten()

            MESSAGE = ""
            MESSAGE = MESSAGE + '300,'
            MESSAGE = MESSAGE + str(1) +','
            MESSAGE = MESSAGE + str(tvec[0]) + ','+ str(tvec[2]) + ',' + str(-tvec[1]) + ','
            MESSAGE = MESSAGE + str(quaternion[0]) + ',' + str(quaternion[2]) + ','+ str(-quaternion[1]) + ',' + str(quaternion[3]) + ','
            MESSAGE = MESSAGE + '0.15,'  + '0.15,' +'300'
            sock.sendto(bytes(MESSAGE, "utf-8"), (UDP_IP, UDP_PORT))
            print(tvec, quaternion)


            # Draw Axis
            # cv2.aruco.drawFrameAxes(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)  

    return frame

if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-k", "--K_Matrix", required=True, help="Path to calibration matrix (numpy file)")
    ap.add_argument("-d", "--D_Coeff", required=True, help="Path to distortion coefficients (numpy file)")
    ap.add_argument("-t", "--type", type=str, default="DICT_ARUCO_ORIGINAL", help="Type of ArUCo tag to detect")
    args = vars(ap.parse_args())

    
    if ARUCO_DICT.get(args["type"], None) is None:
        print(f"ArUCo tag type '{args['type']}' is not supported")
        sys.exit(0)

    aruco_dict_type = ARUCO_DICT[args["type"]]
    calibration_matrix_path = args["K_Matrix"]
    distortion_coefficients_path = args["D_Coeff"]
    
    k = np.load(calibration_matrix_path)
    d = np.load(distortion_coefficients_path)

    video = cv2.VideoCapture(0)
    time.sleep(2.0)

    while True:
        ret, frame = video.read()

        if not ret:
            break
        
        output = pose_esitmation(frame, aruco_dict_type, k, d)

        cv2.imshow('Estimated Pose', output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    video.release()
    #cv2.destroyAllWindow()