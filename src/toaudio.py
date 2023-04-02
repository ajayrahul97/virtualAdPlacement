from moviepy.editor import VideoFileClip, concatenate_videoclips

input_video_path = 'k5.mp4'
output_video_path = 'KG_2_ads_national.avi'
final_output_path = 'KG_2_ads_national.mp4'

# Process the video using OpenCV (your code)

# ...

# Release resources
#cap.release()
#cap2.release()
#output_video.release()
#cv2.destroyAllWindows()

# Copy the audio from the input video to the output video using moviepy
input_video = VideoFileClip(input_video_path)
output_video = VideoFileClip(output_video_path)

# Set the output video's audio to the input video's audio
output_video_with_audio = output_video.set_audio(input_video.audio)

# Save the final output video with audio
output_video_with_audio.write_videofile(final_output_path, codec='libx264', audio_codec='aac')

# Close the moviepy video clips
input_video.close()
output_video.close()
output_video_with_audio.close()

