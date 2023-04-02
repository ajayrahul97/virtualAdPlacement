import cv2

input_video_path = 'localnoodle.mp4'
output_video_path = 'local_noodle.mp4'

# Open the input video
cap = cv2.VideoCapture(input_video_path)

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Calculate the number of frames for a 6-second video
total_frames = fps * 6

# Create the VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Read and process frames from the input video
frame_count = 0
while frame_count < total_frames:
    ret, frame = cap.read()
    if not ret:
        break

    # Write the frame to the output video
    output_video.write(frame)
    frame_count += 1

# Release resources
cap.release()
output_video.release()

