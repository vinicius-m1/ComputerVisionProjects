"""
    Simple code used as first experience with OpenCV and Computer Vision. 
    Applies thresholding and circle detection then draw lines linking detected points.

    Author: vinicius-m1
"""

import cv2
import numpy as np

def pipeline(frame, binary_thresh_level, reduce_noise=True):

    original_frame = frame
    
    # grey scale        
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    

    if reduce_noise:
        frame = cv2.GaussianBlur(frame, (5, 5), 0)
        frame = cv2.medianBlur(frame, 5)
        #frame = cv2.medianBlur(frame, 5)
        
    frame = cv2.blur(frame, (3, 3))     
        
    #frame = detect_color(frame)    
        
    #ret, frame = cv2.threshold(frame, 45, 255, cv2.THRESH_BINARY)
    # below threshold to zero
    #ret, frame = cv2.threshold(frame, 250, None, cv2.THRESH_TOZERO_INV)
    
    
    #ret, frame = cv2.threshold(frame, 20, 255, cv2.THRESH_BINARY)
    
    ret, frame = cv2.threshold(frame, None, 190, cv2.THRESH_OTSU)
    
        #cool effect 
    frame = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                           cv2.THRESH_BINARY, 11, 2)
    
    #ret, frame = cv2.threshold(frame, None, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
    
    frame, masked, threshed = get_circles(frame, original_frame)
    
    return frame, original_frame, masked, threshed;


def get_circles(frame, original_frame=None):

    minDist = 60 #20
    param1 = 30 #30
    param2 = 28 #16 #smaller value-> more false circles
    minRadius = 15 #10
    maxRadius = 65 #60
    
    circles = cv2.HoughCircles(frame, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
    
    masked = np.zeros_like(original_frame)
    if circles is not None:
    
        prev = None 
        prev2 = None
        mask = np.zeros_like(original_frame) #canvas like 'original_frame'
        circles = np.uint16(np.around(circles))
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) #so the frame can have colored outlines
        
        for (x,y,r) in circles[0,:]:
            cv2.circle(original_frame, (x,y), r, (0, 0, 255), 3) #draw borders
            cv2.circle(original_frame, (x,y), 2, (0, 0, 255), 3) # draw center
            
            #write coord and radius
            cv2.putText(original_frame, f"x:{x} y:{y}", (x,y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,9,9), 2)
            cv2.putText(original_frame, f"r:{r}", (x,y+40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (9,255,9), 2)
            
            #draw lines to form triangle
            if prev is not None and prev2 is not None:
                cv2.line(original_frame,(x,y),prev,(255,0,0),3)
                cv2.line(original_frame,prev,prev2,(255,0,0),3)
                cv2.line(original_frame,prev2,(x,y),(255,0,0),3)
                
                #area on the middle
                area = heron_formula(prev2,prev,(x,y))
                coord = (int((prev2[0]+prev[0]+x)/3),int((prev2[1]+prev[1]+y)/3)) #calc middle of triangle
                cv2.putText(original_frame, f"A:{int(area)}", coord, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (9,9,255), 2)
                
            prev2 = prev    
            prev = (x,y)
            
            
            if original_frame is not None:
                cv2.circle(mask, (x, y), r, (255, 255, 255), thickness=-1) #to all white(1) inside circle
                masked = cv2.bitwise_and(original_frame, mask) # 0 =nothing, 1= content
            
                
                #cv2.circle(original_frame, (x,y), r, (0, 0, 255), 3) #draw borders
                #cv2.circle(original_frame, (x,y), 2, (0, 0, 255), 3) # draw center
                
    return (original_frame, masked, frame)
        
    
def heron_formula(p1,p2,p3):
    import math
    
    p1 = (float(p1[0]),float(p1[1]))
    p2 = (float(p2[0]),float(p2[1]))
    p3 = (float(p3[0]),float(p3[1]))

    A = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    B = math.sqrt((p3[0] - p2[0])**2 + (p3[1] - p2[1])**2)  
    C = math.sqrt((p1[0] - p3[0])**2 + (p1[1] - p3[1])**2)
    S = (A+B+C)/2
    
    area = math.sqrt(S*(S-A)*(S-B)*(S-C))
    return area

    
def detect_color(frame):

    #convert color modo of frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lowerLimit,upperLimit = (0,0,0), (179,255,30)
    
    mask = cv2.inRange(frame, lowerLimit, upperLimit)
    
    return (mask)
    

camera_mode = False
if camera_mode:
    camera = cv2.VideoCapture(0) #webcam
else: 
    camera = cv2.VideoCapture("video.mp4") #video file as input

frame_width = int(camera.get(3))
frame_height = int(camera.get(4))


out = cv2.VideoWriter('output_video.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 24, (int(frame_height/2), int(frame_width/2)))

binary_thresh_level = 0

while True:

    #frame is a numpy array
    ret, frame = camera.read()
    
    # to horizontal orientation
    
    if not camera_mode: 
        # to horizontal and smaller
        frame = cv2.transpose(frame)[::-1]
    
        frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
    
    #alternate canvas
    #alt_frame = np.zeroes(frame.shape, np.uint8)
    
    
    processed_frame, original_frame, masked, threshed  = pipeline(frame, binary_thresh_level)
    
    #cv2.imshow('Video Output', processed_frame) #shows on frame on window
    #cv2.imshow('Original frame', original_frame)
    #cv2.imshow('Masked frame', masked)
    
    if not camera_mode: 
        frames_grid = np.hstack((original_frame, threshed))
        cv2.imshow('all outputs', frames_grid)
        #cv2.imshow('masked', masked)
        #out.write(processed_frame)
        
    else:
        cv2.imshow('Video Output', processed_frame)
    
    #cv2.imshow('Original', frame) 
    
    #captures keyboard
    if cv2.waitKey(1) == ord('c'):
        binary_thresh_level = binary_thresh_level+5
        print (binary_thresh_level)

    elif cv2.waitKey(1) == ord('v'):
        binary_thresh_level = binary_thresh_level-5
        print (binary_thresh_level)
        
    elif cv2.waitKey(1) == ord('x'):
        break
        
#let go control of cam and close window        
camera.release()    
out.release()    
cv2.destroyAllWindows()

