from AuxillaryFunctions import *
import cv2, time

W, H = 720 , 480
PID = [0.1937, 0.2039, 0]
startDrone = 1
pError_x = 0
pError_y = 0
# Initialize the drone
dronik = telloInit()
#dronik.land()
frm = frameGrabber(dronik, W, H)
while True:
    if startDrone == 1:
        time.sleep(5)
        dronik.takeoff()
        startDrone = 0
    # Grab frame data
    frm = frameGrabber(dronik, W, H)
    # Find the face
    frm, result = faceFinder(frm)
    # Get data for drone
    pError_x, pError_y, frm = trackingFace(dronik, result, W, H, PID, pError_x, pError_y, frm)
    #print(f'X: {result[0][0]} ,Y:{result[0][1]}')
    cv2.imshow('Image', frm)
    if cv2.waitKey(2) & 0xFF == ord('q'):
        dronik.land()
