import cv2
import numpy as np
#from google.colab.patches import cv2_imshow


# set up video capture
cap = cv2.VideoCapture('KarthikPodcast.mp4')
#cap = cv2.VideoCapture('video5.mp4')
#cap = cv2.VideoCapture('video2.mp4')

# initialize variables
rectangles = []
last_detected = {}
max_rectangles = []
duration = 0

while True:
    # read frame
    ret, frame = cap.read()

    if not ret:
        #print("k")
        break

    # convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # threshold image
    _, thresh = cv2.threshold(gray, 150, 250, cv2.THRESH_BINARY)
    kernel = np.ones((5,5), np.uint8)
    eroded = cv2.erode(thresh, kernel, iterations=1)
    dilated = cv2.dilate(eroded, kernel, iterations=1)

    # find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    area2 = 0
    cnts = sorted(contours, key = cv2.contourArea, reverse = True)[:5]
    # iterate through contours
    for contour in cnts:
        # approximate contour as a polygon
        approx = cv2.approxPolyDP(contour, cv2.arcLength(contour, True) * 0.02, True)
        #print(approx)

        # check if polygon has 4 sides and is convex
        if len(approx) == 4 and cv2.isContourConvex(approx):
            # calculate area of polygon
            area = cv2.contourArea(approx)
            if max_area < area:
              max_area = area
              area2 = approx


            # discard polygons with small areas
            if area > 1000:
                # get bounding rectangle of polygon
                x, y, w, h = cv2.boundingRect(approx)
                if w/h > 1.5 :
                    continue

                # add rectangle to list of rectangles
                rectangles.append((x, y, w, h))

    # iterate through rectangles
    for rect in rectangles:
        # check if rectangle has been detected before
        if rect in last_detected:
            # increment duration of detection
            last_detected[rect] += 1
        else:
            # add new detection
            last_detected[rect] = 1

        # check if rectangle has persisted for at least 5 seconds
        if last_detected[rect] >= 10:
            # add rectangle to list of max rectangles
            max_rectangles.append(rect)

    # display rectangles
    #print(rectangles)
    #rect = sorted(max_rectangles)
    #print(rect)
    #x, y, w, h = rect
    #cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
    #cv2.drawContours(frame, [approx], (0,0,255), 2)
    #print(max_rectangles)

    for rect in max_rectangles:
        x, y, w, h = rect
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        #break
        #cv2.drawContours(frame, [approx], (0,0,255), 2)


    # display frame
    cv2.namedWindow("Resized_Window", cv2.WINDOW_NORMAL)

    cv2.resizeWindow("Resized_Window", 600, 600)

    cv2.imshow('Resized_Window', frame)

    # check for key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#print(max_area)
#print(area2)
# release video capture and close windows
# display rectangles
#cv2.rectangle(frame, (808, 435), (808+100, 849+100), (0, 0, 255), 2)
# display frame
#cv2.imshow('frame', frame)

# check for key press
#if cv2.waitKey(1) & 0xFF == ord('q'):
#    break

cap.release()
cv2.destroyAllWindows()

cap2 = cv2.VideoCapture('KarthikPodcast.mp4')
#cap2= cv2.VideoCapture('video5.mp4')
img = cv2.VideoCapture('lutron_logo.mp4')

# Get Image dimensions
img.set(cv2.CAP_PROP_FRAME_WIDTH, 450)  # float `width`
img.set(cv2.CAP_PROP_FRAME_HEIGHT, 350)
width = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap2.get(cv2.CAP_PROP_FPS))
#width = 700
#height = 700
output_video = cv2.VideoWriter('KG_2_ads_national.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))


#image =cv2.imread("test3.jpeg")
#h = image.shape[0]
#w= image.shape[1]
#print(h)
#w = image.set(cv2.CAP_PROP_FRAME_WIDTH, 150)  # float `width`
#h = image.set(cv2.CAP_PROP_FRAME_HEIGHT, 150)
video_frame_counter = 0
tempx = max_rectangles[0][0]
tempy = max_rectangles[0][1]
w,h = max_rectangles[video_frame_counter][2], max_rectangles[video_frame_counter][3]
while True:
    
    ret, frame = cap2.read()
    if not ret:
        print("k")
        break
    ret_video, frame_video = img.read()
    video_frame_counter += 1

    #if video_frame_counter == img.get(cv2.CAP_PROP_FRAME_COUNT):
    #    video_frame_counter = 0
     #   img.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if ret_video:
            # add image to frame
            #frame_video = cv2.resize(image, (200, 200))
    #frame[740:740 + 250, 1896:1896 + 250] = frame_video
        
        if (abs(max_rectangles[video_frame_counter][0] - tempx > 100) and abs(tempy - max_rectangles[video_frame_counter][1]) > 100):
            tempx = max_rectangles[video_frame_counter][0]
            tempy = max_rectangles[video_frame_counter][1]   
            w,h = max_rectangles[video_frame_counter][2], max_rectangles[video_frame_counter][3]
        #print(w,h)
        
        frame_video = cv2.resize(frame_video, (w, h))
  
        frame[tempy:tempy+ h,tempx:tempx + w, :] = frame_video #image[0:169,0:24, :]




    #dst = cv2.addWeighted(frame,0.5,frame_video,0.7,0)

    # display frame
    cv2.namedWindow("Resized_Window", cv2.WINDOW_NORMAL)

    cv2.resizeWindow("Resized_Window", 600, 600)
    output_video.write(frame)

    cv2.imshow('Resized_Window', frame)
    #cv2.imshow('frame',frame)
    # check for key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

img.release()
cap2.release()
cv2.destroyAllWindows()

''' pts_dst = np.array([[1896, 740], [1920,740], [1896,909],[1920,909]])

    pts_src = np.array([[0,0],[0,h-1], [w-1, h-1],[w-1, 0]])

    pt, status = cv2.findHomography(pts_src, pts_dst)

    arrows = cv2.warpPerspective(image, pt, (frame.shape[1],frame.shape[0]))
    #frame[y:y + width, x:x + height] = frame_video
    #window_name='Projector'
            # cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
            # cv2.moveWindow(window_name, screen.x - 1, screen.y - 1)
            # cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
            #                   cv2.WINDOW_FULLSCREEN)'''
