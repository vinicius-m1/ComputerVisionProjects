"""
    Code used to experiment with morphological operations, focused on morphological segmentation.
    ps: utilizes thinning algorithm from contrib version of OpenCV.

    Author: vinicius-m1
"""

# TEST FOR MORPHOLOGICAL ITERATIONS https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
import  cv2
import numpy as np
import sys
kernel = np.ones((5,5),np.uint8)
num_morph_runs = 0

radius = 4
size = 2 * radius + 1  #size must be odd

circle_form = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))

#original_cells_image = cv2.imread('output_image_filters.jpg')


def run_contours_segmentation(image):
    #global original_cells_image
    edges = cv2.Canny(image, 70, 275)  # Adjust thresholds as needed    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_contours = 0
    num_clusters = 0
    color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for contour in contours:
        num_contours += 1
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        cv2.drawContours(color_image, contour, -1, (0, 255, 0), 3)
        
        if area > 5500: 
            num_clusters +=1
            
            # creates image with isolated cluster contours
            mask = np.zeros_like(image)
            cv2.drawContours(mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)
            white_background = np.ones_like(image) * 255
            cluster_image = cv2.bitwise_and(white_background, mask)
            
            #if True:
            #    full_color_cluster = np.zeros_like(original_unaltered)
            #    full_color_cluster[mask == 255] = original_unaltered[mask == 255]
            #    cv2.imwrite(f"steps/cluster{num_clusters}_0_{sys.argv[1]}", full_color_cluster)
            #    cv2.imshow('full color cluster', full_color_cluster)
            
            #passes to morphological segmentation function
            counted, num_cells = automated(cluster_image, num_clusters)
            if counted is None: continue#skip this cluster
            print (f"[i] Automated gave {num_cells} cells")  
            
            center_x = (x+w // 2)
            center_y = (y+h // 2)
                
                
            cv2.putText(color_image, f"{num_cells}", (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)       
    cv2.imshow('numered', color_image)
    
    return image



def automated(cluster, num_clusters=0, full_color_cluster=None):
    
    step = 0
    while True: 
        step += 1
        edges = cv2.Canny(cluster, 70, 275) 
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        prev_amount = len(contours)
        #if its here its usually more than 1 cell
        prev_image = cluster
        
         
        cluster = run_morphological_big(cluster) #shrink
        
        #check for small useful data
        #edges = cv2.Canny(cluster, 70, 275) 
        #contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  
              
        #small = False      
        #for contour in contours:
        #    if cv2.contourArea(contour) < 25 and cv2.contourArea(contour) > 15:
        #        cluster = run_morphological_small(cluster) #increase
        #        cv2.imshow('increased', cluster)
        #        small = True
        #        break        
        
        #cluster  = run_medium_blur(cluster) #clearer image
        cluster = run_opening_closing(cluster) #clearer image
        
        #check result (number of cells)
        edges = cv2.Canny(cluster, 70, 275) 
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
        
        
        
        print(f" - Number of detected cells: {len(contours)}")  
        
        #cv2.imshow('steps', cluster)
        #cv2.waitKey(0)
        
        #if True:
        #    cv2.imwrite(f"steps/cluster{num_clusters}_{step}_{sys.argv[1]}", cluster)
             
        if len(contours) == (prev_amount):
            continue #goes to more shrink
        elif len(contours) > (prev_amount):
            continue
        elif len(contours) < (prev_amount): #messed up, old values will be used
            #cv2.imshow('result automation', prev_image)
            print(f"[+] Result of automation was: {prev_amount}")    
            return prev_image, prev_amount
        elif len(contours) == 0:
            print("[!] Error in automation.")
            return None, None



def run_opening_closing(image):
    kernel1 = np.ones((3, 3), np.uint8)
    kernel2 = np.ones((3, 3), np.uint8)
    processed_frame = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel2)    
    processed_frame = cv2.morphologyEx(processed_frame, cv2.MORPH_CLOSE, kernel1) 
    print("[i] Ran opening+closing.")   
    return processed_frame

def run_distance(image):
    dist_transform = cv2.distanceTransform(image, cv2.DIST_L2, 5)
    print ("[i] Ran distance transform.")
    return dist_transform

def run_adaptative_threshold(image):
    print (f"[i] applied [adaptative] threshold")
    thresholded = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return thresholded

def run_reset():
    print ("[!] Reset. -----------")
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

def run_thinning(image):
    print (f"[i] Applied thinning")
    thinned = cv2.ximgproc.thinning(image)  
    return thinned

def run_medium_blur(image):
    print (f"[i] Applied medium blur")
    median_blurred_image = cv2.medianBlur(image, 3)
    return median_blurred_image

def run_contours(image):
    edges = cv2.Canny(image, 70, 275)  # Adjust thresholds as needed    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_contours = 0
    num_clusters = 0
    for contour in contours:
        num_contours += 1
        #x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        #cv2.drawContours(image, contour, -1, (0, 255, 0), 6)
        
        if area > 4000: 
            num_clusters +=1
            if num_clusters < 9:
                mask = np.zeros_like(image)
                cv2.drawContours(mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)
                white_background = np.ones_like(image) * 255
                cluster_image = cv2.bitwise_and(white_background, mask)
                
                cv2.imwrite(f'cluster_{num_clusters}.png', cluster_image)
            #center_x = (x+w // 2) # // devides as integer
            #center_y = (y+h // 2)    
            #cv2.circle(image, (center_x,center_y), 1, (0,0,255), 10)    
    print (f"[i] Contours: {num_contours} ... Clusters: {num_clusters}")
    
    return image

def run_morphological_big(image):

    erosion = cv2.erode(image, circle_form, iterations=1) 
    global num_morph_runs
    num_morph_runs +=1
    print (f"[i] Morph operations(big) ran: {num_morph_runs}")    
    return erosion

def run_binary_threshold(image, threshold_value=60):
    print (f"[i] applied threshold {threshold_value}")
    _, thresholded = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
    return thresholded

def run_morphological_small(image):
    erosion = cv2.dilate(image,circle_form,iterations = 1)    

    global num_morph_runs
    num_morph_runs +=1
    print (f"[i] Morph operations(small) ran: {num_morph_runs}")    
    return erosion


def main(image):

    processed_image = None
    key = cv2.waitKey(10)

    if key == ord('m'):
        processed_image = run_morphological_small(image)
    elif key == ord('n'):
        processed_image = run_morphological_big(image)    
    elif key == ord('t'):
        processed_image = run_thinning(cv2.bitwise_not(image))     
    elif key == ord('d'):
        processed_image = run_distance(image)     
    elif key == ord('h'):
        processed_image = run_binary_threshold(image)    
    elif key == ord('r'):
        processed_image = run_reset() 
    elif key == ord('c'):
        processed_image = run_contours(image)   
    elif key == ord('o'):
        processed_image = run_opening_closing(image)         
    elif key == ord('w'):
        processed_image, _ = automated(image)
    elif key == ord('s'):
        processed_image = run_contours_segmentation(image)                   
    elif key == ord('a'):    
        processed_image = run_adaptative_threshold(image)         
    elif key == ord('k'):    
        processed_image = run_medium_blur(image)                                          
    else:
        processed_image = image    
    
    return processed_image    
      
image_path = sys.argv[1]
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
original_unaltered = cv2.imread(image_path)
print("[i] Image Loaded.")    


# run updating with dynamic input
while True:      
    image = main(image)
    cv2.imshow('output', image)      
cv2.destroyAllWindows()    
