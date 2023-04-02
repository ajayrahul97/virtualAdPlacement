import cv2
import numpy as np

cap = cv2.VideoCapture('k5.mp4')
res = {}
f = 0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    f += 1
    # Preprocessing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
    
    # Edge detection
    #edges = cv2.Canny(thresh, 50, 150, apertureSize=3)
    edges = cv2.Canny(blur, 1, 2000)
    # Contour detection
    # convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # threshold image
    _, thresh = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)

    # find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by size
    for cnt in contours:
            
        # Approximate contours to polygons
        approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)
        
        # Filter polygons by number of sides
        if len(approx) < 4:
            continue
            
        # Filter polygons by angle between sides
        (x, y, w, h) = cv2.boundingRect(approx)
        if w*h < 12000 or w*h > 100000:
            continue
        if x < 0.1 * width:
            continue
       	if len(res) == 0:
            res[(x,y,w,h,f)] = 1
        else:
            is_f = False
            for a,b,c,d,f2 in res:
                if (x > 0.8*a and x < 1.2*a) and (y > 0.8*b and y < 1.2*b) and (w > 0.8*c and w < 1.15*c) and (h > 0.8*d and h < 1.2*d):
                    res[(a,b,c,d,f2)] += 1
                    is_f = True
            if is_f == False:
                 res[(x,y,w,h,f)] = 1
        # Draw rectangles
        cv2.rectangle(frame, (x,y),(x+w, y+h),(0, 255, 0), 2)
        print(x,y,w,h, w*h)
        # break
    # Display the resulting frame
    cv2.imshow('frame', frame)
    
    # Exit on key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


print(res)

# iv2.destroyAllWindows()
cap = cv2.VideoCapture('k5.mp4')
# cap2 = cv2.VideoCapture('k3.mp4')
alpha = 0.1

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
output_video = cv2.VideoWriter('KG_2_ads_national.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height))
f = 0
cc = 0
while True:
    ret1, frame1 = cap.read()
    if not ret1:
        break
    f += 1
    is_t = False
    for a,b,c,d,f2 in res:
        if f2 != f:
            continue
        if res[(a,b,c,d,f2)] < 150:
            continue
        cap2 = None
        if cc == 0:
            cap2 = cv2.VideoCapture('local_noodle.mp4')
            cc = 1
        else:
            cap2 = cv2.VideoCapture('local_music.mp4')
            cc = 0
        while True:
            ret2, frame2 = cap2.read()
            if not ret2:
                break

            # Resize the second video's frame to the specified width and height
            print(c,d)
            resized_frame2 = cv2.resize(frame2, (c + int(0.1 * c), d + int(0.1 * d)))
            # Overlay the resized second video's frame on the first video's frame using the specified coordinates
            print(b, b+d, a, a+c,width, height)
            roi = frame1[b:b + d + int(0.1 * d), a:a + c + int(0.1 * c)]
            blended_roi = cv2.addWeighted(roi, alpha, resized_frame2, 1 - alpha, 0)
            frame1[b:b + d + int(0.1 * d), a:a + c + int(0.1 * c)] = blended_roi

            # Write the frame with the overlaid video to the output video
            output_video.write(frame1)

            # Display the resulting frame
            cv2.imshow('Blended Video', frame1)
            ret, frame1 = cap.read()
            if not ret:
                break
            f +=1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    if not ret:
        break
    output_video.write(frame1)

    # Display the resulting frame
    cv2.imshow('Blended Video', frame1)



cap.release()
cap2.release()
output_video.release()
cv2.destroyAllWindows()
