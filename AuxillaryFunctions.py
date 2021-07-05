from djitellopy import Tello
import cv2
import numpy as np


def telloInit():
    '''
    Initialize the drone, set velocities to 0, display battery percentage, restart the stream
    :returns initialized Drone object
    '''

    # Initialize the drone
    dronik = Tello()
    dronik.connect()
    dronik.for_back_velocity = 0
    dronik.left_right_velocity = 0
    dronik.up_down_velocity = 0
    dronik.yaw_velocity = 0
    dronik.speed = 0

    # Get battery values in console
    print(f'Amount of battery left {dronik.get_battery()} %')

    # Restart the stream
    dronik.streamoff()
    dronik.streamon()

    return dronik


def frameGrabber(dronik, w=360, h=240):
    '''
    Function that grabs a frame from Tello camera and returns it resized.
    :param dronik: instance of Tello object
    :param w: width of output frame
    :param h: heigth of output frame
    :return: resized frame
    '''
    frame = dronik.get_frame_read()
    frame = frame.frame
    frame_resized = cv2.resize(frame, (w, h))

    return frame_resized


def faceFinder(frm):
    """
    :param frm: frame from drone camera
    :return: frm - the same frame with rectangles on main face detected
    """
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    imgGray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
    facesDetected = faceCascade.detectMultiScale(imgGray, 1.2, 4)
    facesCenter = []
    facesArea = []

    for (x, y, w, h) in facesDetected:
        cv2.rectangle(frm, (x, y), (x + w, y + h), (0, 128, 0), 3)
        facesCenter.append([x + w // 2, y + h // 2])
        facesArea.append(w * h)

    if len(facesArea) != 0:
        i = facesArea.index(max(facesArea))
        return frm, [facesCenter[i], facesArea[i]]
    else:
        return frm, [[0, 0], 0]

def dataDisplay(img, error_x, error_y, spd_x, spd_y, PID):
    '''Take the positional, speed and pid data and display it on video frame'''
    cv2.putText(img, f'PID SETTINGS: Kp:{PID[0]} Ki: {PID[1]} Kd: {PID[2]} ',
                (30,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,0,255),2)
    cv2.putText(img, f'X ERROR: {error_x} X SPEED: {spd_x}',
                (30,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,255,0), 2)
    cv2.putText(img, f'Y ERROR: {error_y} Y SPEED: {spd_y}',
                (30,90), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,255,0),2)
    return img


def trackingFace(dronik, result, w, h, PID, pError_x, pError_y, frm):
    '''

    :param dronik: instance of Tello Object
    :param info: 0 for face not found
    :param w: width
    :param pid: pid controller parameters
    :param dist: distance from where we want to be and where we are camerawise
    :return: the instructions for drone movement
    '''

    #Calculate the velocities based on error values and PID
    error_x = result[0][0] - w // 2
    spd_x = PID[0] * error_x + PID[1] * (error_x - pError_x)
    spd_x = np.clip(spd_x, -100, 100)

    error_y = h//2 - result[0][1]
    spd_y = PID[0] * error_y + PID[1] * (error_y - pError_y)
    spd_y = np.clip(spd_y, -100, 100)
    spd_x, spd_y = int(spd_x), int(spd_y)

    #Display the data on the frame
    frm = dataDisplay(frm, error_x, error_y, spd_x, spd_y, PID)


    #print(f'X SPD: {spd_x}, Y SPD: {spd_y}')

    #Send the values to the tello flight control
    if result[0][0] != 0 or result[0][1] != 0:
        dronik.yaw_velocity = int(spd_x)
        dronik.up_down_velocity = int(spd_y)
    else:
        dronik.for_back_velocity = 0
        dronik.left_right_velocity = 0
        dronik.up_down_velocity = 0
        dronik.yaw_velocity = 0
        error_x = 0
        error_y = 0

    if dronik.send_rc_control:
        dronik.send_rc_control(dronik.left_right_velocity,dronik.for_back_velocity, dronik.up_down_velocity,
                               dronik.yaw_velocity)


    return error_x, error_y, frm
