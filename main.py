"""
Python script which takes in a video (or series of videos) and masks out sections of the video(s) so that only one person is visible at a time.
Meant for use with markerless motion capture systems which can only deal with individual people at a time.
Meant for use with videos with a static camera in which there is a clear boundary between the people in the videos.
"""
import os
import argparse
import cv2
from ultralytics import YOLO
import numpy as np

model = YOLO("yolov8n.pt")


mask_blur_size = None
mask_type = None
fps = None
frame_limit = None


def process_video(video_file_path: str, output_parent_direcory: str):
    """
    Process a single video file.
    """
    print('Processing video: ' + video_file_path)

    # Create the output parent directory if it doesn't exist
    if output_parent_direcory is not None:
        if not os.path.exists(output_parent_direcory):
            os.makedirs(output_parent_direcory)
    
    # Create output subdirectory for this video
    output_dir = os.path.join(output_parent_direcory, os.path.basename(video_file_path))
    output_dir = os.path.splitext(output_dir)[0]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        print(f'Output directory {output_dir} already exists. Please change the output directory or delete the existing directory.')
        return
    

    # Open the video file
    cap = cv2.VideoCapture(video_file_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print(f"FPS: {fps}, frame size: {frame_size}")

    video_writers = []
    max_persons = 0

    frame_count = 0
    # Read the frames
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break
            
        frame_count += 1
        # Stop processing if we've reached the frame limit
        if frame_limit is not None and frame_count > frame_limit:
            break

        results = model(frame)[0]
        print(f"Processing frame {frame_count}")

        # annotated_frame = results.plot()
        # cv2.imshow('frame', annotated_frame)


        # Sort the bounding boxes by their x position so that the ordering remains consistent between frames
        sorted_boxes = sorted(results.boxes, key=lambda box: box.xyxy[0][0])
        max_persons = max(max_persons, len(sorted_boxes))
        print(f"Sorted boxes: {len(sorted_boxes)}")

        # TODO: the bounding boxes sometime jump back to the right edge of the bounding box. Store the previous position and then average it with the current position to smooth out the movement.
        for index, box in enumerate(sorted_boxes):
            frame_copy = frame.copy()
            label = results.names[int(box.cls)]

            box_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

            isolated_person_frame = None

            if label == "person":
                if mask_type == "split":
                    
                    # Calculate the bounds for this box.
                    # The left bound is the midpoint between the this box's left edge and the previous box's right edge.
                    # The right bound is the midpoint between this box's right edge and the next box's left edge.
                    # If this is the first box, the left bound is the left edge of the frame.
                    # If this is the last box, the right bound is the right edge of the frame.
                    left_bound = 0 if index == 0 else int((box.xyxy[0][0] + sorted_boxes[index-1].xyxy[0][2]) / 2)
                    right_bound = frame.shape[1] if index == len(sorted_boxes) - 1 else int((box.xyxy[0][2] + sorted_boxes[index+1].xyxy[0][0]) / 2)

                    # Draw the rectangle on the box_mask
                    x1, y1 = left_bound, 0
                    x2, y2 = right_bound, frame.shape[0]

                    cv2.rectangle(box_mask, (x1, y1), (x2, y2), (255, 255, 255), -1)

                    isolated_person_frame = cv2.bitwise_and(frame_copy, frame_copy, mask=box_mask)
                
                elif mask_type == "box":
                    x1, y1, x2, y2 = (int(coord) for coord in box.xyxy[0])
                    cv2.rectangle(box_mask, (x1, y1), (x2, y2), (255, 255, 255), -1)

                    # Blur the mask to increase the size of the masked area
                    blurred_mask = cv2.GaussianBlur(box_mask, (mask_blur_size, mask_blur_size), 0)

                    # Treshold the image to make it b/w
                    _, expanded_mask = cv2.threshold(blurred_mask, 20, 255, cv2.THRESH_BINARY)

                    # At this point, we have a mask which isolates one person in the frame. Apply the mask to the frame copy.
                    isolated_person_frame = cv2.bitwise_and(frame_copy, frame_copy, mask=expanded_mask)


            # If a writer for this person doesn't exist yet, create it
            if len(video_writers) <= index:
                print(f"Creating video writer for person {index}")
                person_output_path = os.path.join(output_dir, f'person_{index}.mp4')
                video_writer = cv2.VideoWriter(filename=person_output_path, fourcc=cv2.VideoWriter_fourcc(*'mp4v'), fps=fps, frameSize=frame_size, isColor=True)
                video_writers.append(video_writer)
            
            # Add framecount to the frame
            cv2.putText(isolated_person_frame, f"Frame: {frame_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Show the frame
            cv2.imshow('frame', isolated_person_frame)

            # Write the frame to its corresponding video writer
            video_writers[index].write(isolated_person_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break # exit the while loop if the user presses q.

    # Release all video writers
    for video_writer in video_writers:
        video_writer.release()

    cap.release()
    cv2.destroyAllWindows()

    print(f"Processed {frame_count-1} frames.")
    print(f"Max persons: {max_persons}")


def main():
    # Parse the arguments
    parser = argparse.ArgumentParser(description='Mask out sections of a video so that only one person is visible at a time. Outputs a video file for each person detected in the video.')
    parser.add_argument('video', type=str, help='The video or folder of videos to process.')
    parser.add_argument("output", type=str, help="The output directory for the processed video(s). Creates this directory if it doesn't exist.")
    parser.add_argument("--mask-type", type=str, choices=["box", "split"], default="box", help="The type of mask to use. Default is 'box'. 'box' masks using the bounding boxes of each person, 'split' splits the frame vertically between each person.")
    parser.add_argument("--blur-size", type=int, default=71, help="The size of the blur kernel to use when blurring the mask (only used when mask_type is 'box'). Larger values will increase the size of the masked area. Increase this if the mask doesn't cover all of the movement/props the person has.")
    parser.add_argument("--frame-limit", type=int, default=-1, help="The number of frames to process. Useful for testing. If not specified, the entire video will be processed.")

    args = parser.parse_args()

    global mask_blur_size
    mask_blur_size = args.blur_size

    global mask_type
    mask_type = args.mask_type

    if args.frame_limit != -1:
        print(f'Processing {args.frame_limit} frames.')
        global frame_limit
        frame_limit = args.frame_limit

    if args.mask_type not in ['box', 'split']:
        print('Invalid mask type. Must be either "box" or "split".')
        exit()

    # Check if the video (mp4, wav, mkv) is a single video or a folder of videos
    if args.video.endswith('.mp4') or args.video.endswith('.mkv') or args.video.endswith('.wav'):
        print('Single video')

        video_file_path = args.video
        process_video(video_file_path, args.output)

    elif os.path.isdir(args.video):
        print('Folder of videos')
        video_file_paths = [os.path.join(args.video, f) for f in os.listdir(args.video) if os.path.isfile(os.path.join(args.video, f)) and (f.endswith('.mp4') or f.endswith('.mkv') or f.endswith('.wav'))]

        if video_file_paths is None or len(video_file_paths) == 0:
            print('No video files found in the folder.')
            exit()

        for video_file_path in video_file_paths:
            process_video(video_file_path, args.output)

    else:
        print('Invalid video file or folder.')


if __name__ == "__main__":
    main()
