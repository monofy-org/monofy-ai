import cv2
import imageio

def double_frame_rate_with_interpolation(input_path, output_path, max_frames: int = None):
    # Open the video file using imageio
    video_reader = imageio.get_reader(input_path)
    fps = video_reader.get_meta_data()['fps']
    
    # Calculate new frame rate
    new_fps = 2 * fps

    metadata = video_reader.get_meta_data()
    print(metadata)

    # Get the video's width and height
    width, height = metadata['size']
    print(f"Video dimensions: {width}, {height}")

    # Create VideoWriter object to save the output video using imageio
    video_writer = imageio.get_writer(output_path, fps=new_fps)

    # Read the first frame
    prev_frame = video_reader.get_data(0)

    if max_frames is None:
        max_frames = len(video_reader)
    else:
        max_frames = max(max_frames, len(video_reader))

    print(f"Processing {max_frames} frames...")
    for i in range(1, max_frames):        

        try:
            # Read the current frame
            frame = video_reader.get_data(i)

            # Linear interpolation between frames
            interpolated_frame = cv2.addWeighted(prev_frame, 0.5, frame, 0.5, 0)

            # Write the original and interpolated frames to the output video
            video_writer.append_data(prev_frame)
            video_writer.append_data(interpolated_frame)

            prev_frame = frame
        except IndexError as e:
            print(f"IndexError: {e}")
            break

    # Close the video writer
    video_writer.close()

    print(f"Video with double frame rate and interpolation saved at: {output_path}")
