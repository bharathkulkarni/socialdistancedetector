
# imported packages
import cv2
import numpy as np
import time
import argparse

# own modules
import distance_calculator, draw_bounding_box

confid = 0.5
thresh = 0.5
click_pts = []


# Function to get points for Region of Interest(ROI) and distance scale. It will take 8 points on first frame using mouse click    
# event.First four points will define ROI where we want to moniter social distancing. Also these points should form parallel  
# lines in real world if seen from above(birds eye view). Next 3 points will define 6 feet(unit length) distance in     
# horizontal and vertical direction and those should form parallel lines with ROI. Unit length we can take based on choice.
# Points should pe in pre-defined order - bottom-left, bottom-right, top-right, top-left, point 5 and 6 should form     
# horizontal line and point 5 and 7 should form verticle line. Horizontal and vertical scale will be different. 

# Function will be called on mouse events                                                          

def get_clicked_points(event, x, y, flags, frame):

    global click_pts
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(click_pts) < 4:
            cv2.circle(image, (x, y), 5, (0, 255, 0), 10)
        else:
            cv2.circle(image, (x, y), 5, (0, 0, 255), 10)
            
            #draw four lines for all four adjacent clicked points
        if len(click_pts) >= 1 and len(click_pts) <= 3:
            cv2.line(image, (x, y), (click_pts[len(click_pts)-1][0], click_pts[len(click_pts)-1][1]), (70, 70, 70), 2)
            if len(click_pts) == 3:
                cv2.line(image, (x, y), (click_pts[0][0], click_pts[0][1]), (70, 70, 70), 2)
        
        if "click_pts" not in globals():
            click_pts = []
        click_pts.append((x, y))
        
        


def calculate_social_distancing(vid_path, net, output_vid, ln1):
    
    count = 0
    vs = cv2.VideoCapture(vid_path)    

    # Get video height, width and fps
    height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = int(vs.get(cv2.CAP_PROP_FPS))
    
    # Set scale for birds eye view
    # Bird's eye view will only show ROI
    scale_w, scale_h = distance_calculator.get_scale(width, height)

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    output_video = cv2.VideoWriter("./output_vid/distancing.avi", fourcc, fps, (width, height))
    
        
    points = []
    global image
    
    while True:

        (grabbed, frame) = vs.read()

        if not grabbed:   
            break
            
        (H, W) = frame.shape[:2]
        
        # first frame will be used to draw ROI and horizontal and vertical 6 ft distance(unit length in both directions)
        if count == 0:
            while True:
                image = frame
                cv2.imshow("image", image)
                cv2.waitKey(1)
                if len(click_pts) == 8:
                    cv2.destroyWindow("image")
                    break
               
            points = click_pts      
                 
        # first four clicked points are used for bird's eye view transformation. The region marked by these 4 points are 
        # considered ROI. This polygon shaped ROI is then transformed into a rectangle which becomes the bird eye view. 
        src = np.float32(np.array(points[:4]))
        dst = np.float32([[0, H], [W, H], [W, 0], [0, 0]])
        birds_eye_view = cv2.getPerspectiveTransform(src, dst)

        # using next 3 points for horizontal and vertical unit length(in this case 6 ft(180 cm))
        pts = np.float32(np.array([points[4:7]]))
        transformed_pt = cv2.perspectiveTransform(pts, birds_eye_view)[0]
        
        
        # distance_w and distance_h will give us 180 cm distance in both horizontal and vertical directions
        distance_w = np.sqrt((transformed_pt[0][0] - transformed_pt[1][0]) ** 2 + (transformed_pt[0][1] - transformed_pt[1][1]) ** 2)
        distance_h = np.sqrt((transformed_pt[0][0] - transformed_pt[2][0]) ** 2 + (transformed_pt[0][1] - transformed_pt[2][1]) ** 2)
        pnts = np.array(points[:4], np.int32)
        cv2.polylines(frame, [pnts], True, (70, 70, 70), thickness=2)
    

    
        # detection of pedestrians using YOLO v3
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        personID=0
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln1)
        end = time.time()
        boxes = []
        confidences = []
        classIDs = []   
    
        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                # detecting humans in frame
                if classID == personID:

                    if confidence > confid:

                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype("int")

                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))

                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)
                    
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, confid, thresh)
        font = cv2.FONT_HERSHEY_PLAIN
        boxes1 = []
        for i in range(len(boxes)):
            if i in idxs:
                boxes1.append(boxes[i])
                x,y,w,h = boxes[i]
                
        if len(boxes1) == 0:
            count = count + 1
            continue
            
        
        person_points = distance_calculator.get_transformed_points(boxes1, birds_eye_view)
        
        #distance between transformed points(humans)
        distance_matrix, bxs_matrix = distance_calculator.get_distances(boxes1, person_points, distance_w, distance_h)
        violators_count = distance_calculator.get_count(distance_matrix)
    
        frame1 = np.copy(frame)
        
        # draw_bounding_box the frame with bouding boxes around humans which hilight violators    
        
        img = draw_bounding_box.social_distancing_view(frame1, bxs_matrix, boxes1, violators_count)
        
        # Show/write image and videos
        if count != 0:
            output_video.write(img)
            cv2.imshow('frame', img)

    
        count = count + 1
        # if the `q` key is pressed, exit program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
     
    vs.release()
    cv2.destroyAllWindows() 
        

if __name__== "__main__":

    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-v', '--video_path', action='store', dest='video_path', default='./data/example.mp4' ,
                    help='Path for input video')
                    
    parser.add_argument('-O', '--output_vid', action='store', dest='output_vid', default='./output_vid/' ,
                    help='Path for Output videos')

    parser.add_argument('-m', '--model', action='store', dest='model', default='./yolo/',
                    help='Path for models directory')
                    
    parser.add_argument('-u', '--uop', action='store', dest='uop', default='NO',
                    help='Use open pose or not (YES/NO)')
                    
    values = parser.parse_args()
    
    model_path = values.model
    if model_path[len(model_path) - 1] != '/':
        model_path = model_path + '/'
        
    
    output_vid = values.output_vid
    if output_vid[len(output_vid) - 1] != '/':
        output_vid = output_vid + '/'


    # load Yolov3 weights
    
    weightsPath = model_path + "yolov3.weights"
    configPath = model_path + "yolov3.cfg"

    net_yl = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    ln = net_yl.getLayerNames()
    ln1 = [ln[i - 1] for i in net_yl.getUnconnectedOutLayers()]

    # set mouse callback 

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", get_clicked_points)
    np.random.seed(42)
    
    calculate_social_distancing(values.video_path, net_yl, output_vid, ln1)



