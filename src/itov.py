import cv2

image_path = 'lutron-logo.png'
output_video_path = 'lutron_logo.mp4'

# Read the input image
image = cv2.imread(image_path)

# Get the image dimensions
height, width, _ = image.shape

# Calculate the total number of frames for a 6-second video at 30 fps
fps = 30
duration = 6  # seconds
total_frames = fps * duration

# Create the VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Write the image to the output video as many times as needed to create the desired duration
for _ in range(total_frames):
    output_video.write(image)

# Release resources
output_video.release()

