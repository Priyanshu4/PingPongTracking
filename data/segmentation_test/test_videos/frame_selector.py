""" Script to select a single frame from a video.
"""

import cv2
import argparse
import sys

def select_frame(video_path, output_path):
    # Load the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        sys.exit()

    frame_index = 0
    while True:
        # Read a frame
        ret, frame = cap.read()

        # If frame is read correctly, ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Display the current frame
        cv2.imshow('Frame', frame)

        # Wait for key press
        key = cv2.waitKey(0) & 0xFF

        # Map 'q' key to exit
        if key == ord('q'):
            break
        # Map 's' key to save the frame
        elif key == ord('s'):
            save_path = f'{output_path}/frame_{frame_index}.png'
            cv2.imwrite(save_path, frame)
            print(f"Frame {frame_index} saved as {save_path}.")

        # Update frame index
        frame_index += 1

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Frame Selector for Videos')
    parser.add_argument('video_path', type=str, help='Path to the video file')
    parser.add_argument('--output_path', type=str, default='.', help='Directory to save frames')
    
    args = parser.parse_args()

    select_frame(args.video_path, args.output_path)

if __name__ == "__main__":
    main()
