
# imported packages
import cv2
import numpy as np

# Function to calculate bottom center for all bounding boxes and transform prespective for all points.
def get_transformed_points(boxes, prespective_transform):
    
    bottom_points = []
    for box in boxes:
        #get bottom center point of the box
        pnts = np.array([[[int(box[0]+(box[2]*0.5)),int(box[1]+box[3])]]] , dtype="float32")
        #transform bottom center point to bird's eye view
        bd_pnt = cv2.perspectiveTransform(pnts, prespective_transform)[0][0]
        pnt = [int(bd_pnt[0]), int(bd_pnt[1])]
        bottom_points.append(pnt)
        
    return bottom_points

# Function calculates distance between two points(pedestrians). distance_w, distance_h represents number
# of pixels in 180cm length horizontally and vertically. We calculate horizontal and vertical
# distance in pixels for two points and get ratio in terms of 180 cm distance using distance_w, distance_h.
# Then we calculate how much cm distance is horizontally and vertically and then using pythagoras
# we calculate distance between points in terms of cm. 
def cal_dis(p1, p2, distance_w, distance_h):
    
    h = abs(p2[1]-p1[1])
    w = abs(p2[0]-p1[0])
    
    #horizontal distance in bird's eye view
    dis_w = float((w/distance_w)*180)
    #vertical distance in bird's eye view
    dis_h = float((h/distance_h)*180)
    
    return int(np.sqrt(((dis_h)**2) + ((dis_w)**2)))

# Function calculates distance between all pairs if closeness is 0 then there is violation of social distancing.
def get_distances(boxes1, bottom_points, distance_w, distance_h):
    
    distance_matrix = []
    bxs = []
    
    for i in range(len(bottom_points)):
        for j in range(len(bottom_points)):
            if i != j:
                dist = cal_dis(bottom_points[i], bottom_points[j], distance_w, distance_h)
                #dist = int((dis*180)/distance)
                if dist <= 180:
                    closeness = 0
                    distance_matrix.append([bottom_points[i], bottom_points[j], closeness])
                    bxs.append([boxes1[i], boxes1[j], closeness])
                    
                else:
                    closeness = 1
                    distance_matrix.append([bottom_points[i], bottom_points[j], closeness])
                    bxs.append([boxes1[i], boxes1[j], closeness])
                
    return distance_matrix, bxs
 
# Function gives scale for birds eye view               
def get_scale(W, H):
    
    dis_w = 400
    dis_h = 600
    
    return float(dis_w/W),float(dis_h/H)
    
# Function gives count for violators and non violators    
def get_count(distances_mat):
    
    r = []
    g = []
  
    
    for i in range(len(distances_mat)):

        if distances_mat[i][2] == 0:
            if (distances_mat[i][0] not in r) and (distances_mat[i][0] not in g):
                r.append(distances_mat[i][0])
            if (distances_mat[i][1] not in r) and (distances_mat[i][1] not in g):
                r.append(distances_mat[i][1])
                
    return (len(r),len(g))
