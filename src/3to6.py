import cv2

input_video_path = 'giphy.mp4'
output_video_path = 'giphy.mp4'

# Open the input video
cap = cv2.VideoCapture(input_video_path)

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Create the VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Read and process frames from the input video
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Write the frame twice to the output video to duplicate each frame
    output_video.write(frame)
    output_video.write(frame)

# Release resources
cap.release()
output_video.release()

