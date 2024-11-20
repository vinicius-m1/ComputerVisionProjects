"""
    This code utilizes background subtraction to detect moving objects and employs distance-based tracking for object identification. It also incorporates contour detection and line crossing detection to count objects as they cross predefined lines.
    
    Author: vinicius-m1
"""

import cv2
import numpy as np
import math
import sys
cv2.setUseOptimized(True)
line_counts = [0, 0]
tracked_objects = {}

#last frame of each object was counted
last_counted = {}
next_id = 1
frame_count = 0
tracks =[]
select_track = 0

lines = [
    [(168, 309), (216, 460)],
    [(357, 497), (864, 427)]
]

colors = [
    (255, 0, 0),    # Blue
    (0, 255, 0),    # Green
    (0, 0, 255),    # Red
    (255, 255, 0),  # Cyan
    (0, 255, 255),  # Yellow
    (255, 0, 255),  # Magenta
    (255, 165, 0),  # Orange
    (128, 0, 128),  # Purple
    (255, 192, 203), # Pink
    (255, 255, 255), # White
    (0, 0, 0) # Black
]


def draw_lines(frame):
    for i in range(len(lines)):
        line = lines[i];
        
        cv2.line(frame, line[0], line[1], (0, 244, 0), 2)
        
        # get center
        center_x = (line[0][0] + line[1][0]) // 2
        center_y = (line[0][1] + line[1][1])//2
        
        # draw info
        text = f"count: {line_counts[i]}"
        cv2.putText(frame, text, (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# get relative positions to the counting line
def is_counterclockwise(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def do_intersect(A, B, C, D):
    #if the two ends are in oposite sides of the line (crossed)
    return (is_counterclockwise(A, C, D) != is_counterclockwise(B, C, D) and is_counterclockwise(A, B, C) != is_counterclockwise(A, B, D))

def get_distance(point1, point2):
    x1,y1 = point1
    x2,y2 = point2
    return math.sqrt( (x2-x1)**2 +(y2-y1)**2 )

def get_rand_color():
    return(colors[np.random.randint(0, (len(colors)-1))])
    

def pipeline(frame):
    global tracked_objects, last_counted, next_id, frame_count, tracks

    # background subtraction
    mask = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = bg_subtractor.apply(frame)

    kernel_small = np.ones((3, 3), np.uint8)
    kernel_large = np.ones((11, 11), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large)

    # get contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    #cv2.imshow('Video1', mask)
    
    
    current_objects = {}
    current_tracks = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if 500 < area < 100000:
            x, y, w, h = cv2.boundingRect(contour)
            center_x = (x+w // 2) # // devides as integer
            center_y = (y+h // 2)
           

            # TRACKING PART
            object_id = next_id
            #identify the obeject
            for id in tracked_objects:
                prev_x, prev_y, color = tracked_objects[id] # get previous coord
                #if movemente between frames less than 50 consider the same object 
                distance = get_distance((center_x, center_y), (prev_x, prev_y))
                if distance < 50:
                    object_id = id
                    #if frame_count%5 == 0:
                    tracks.append(((center_x, center_y), (prev_x, prev_y), color, object_id ))
                    break
                    
            # draw tracks    
            for track in tracks:
                start_pos, end_pos, color, obj_id = track
                #if select_track != 0 or (obj_id not in current_objects):
                if select_track != 0 :
                    if obj_id != select_track: continue
                elif len(tracks) > 500:
                    tracks.pop(0)          
                              
                cv2.line(frame, start_pos, end_pos, color, 3)                            
                        
            
            # if not identified, created new id
            if object_id == next_id:
                next_id += 1
                color = get_rand_color()
                

            current_objects[object_id] = (center_x, center_y, color)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            #COUNTING PART
            for i in range(len(lines)):
                line = lines[i]
                
                #if tracked object insersects with one of the lines
                if (object_id in tracked_objects and do_intersect(tracked_objects[object_id], (center_x, center_y), line[0], line[1])):                                 #old coord, new coord, line coords
                    
                    # check if not already counted or counted long ago(last 30 frames)
                    if object_id not in last_counted or frame_count - last_counted[object_id] > 30:
                        line_counts[i] += 1
                        last_counted[object_id] = frame_count #sets last time this object was counted(now)

            
            cv2.putText(frame, f"ID: {object_id}", (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    #update
    tracked_objects = current_objects
    #tracks = current_tracks

    # draw lines
    draw_lines(frame)
    #print (tracked_objects)
    # draw info
    total_count = line_counts[0] + line_counts[1]
    cv2.putText(frame, f'Total Count: {total_count}', (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    cv2.putText(frame, f'Current Objects: {len(current_objects)}', (8, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return frame

try:
    video = cv2.VideoCapture(sys.argv[1])
except:
    print("[!] Error loading video. (pass as argument)")
    sys.exit(1)

bg_subtractor = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400, detectShadows=True)

#bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=60, varThreshold=50)

while True:
    ret, frame = video.read()
    if not ret:
        break
    
    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    frame_count += 1
    
    processed_frame = pipeline(frame)
   
    cv2.imshow('Video', processed_frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('n'):
        select_track += 1
        print (f"obj_id = {select_track}")
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Clean up
video.release()     
cv2.destroyAllWindows()
