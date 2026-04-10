import os
import re
import cv2


def frames_to_video(frames_dir, fps=10, video_name="video.mp4"):
    """
    Creates a video from numbered frames in a folder and saves it in the same folder.

    Parameters
    ----------
    frames_dir : str
        Directory containing frames (e.g. topdown_0_36.jpg)
    fps : int
        Frames per second of the output video
    video_name : str
        Name of the output video file
    """

    # Collect frame files
    frame_files = [
        f for f in os.listdir(frames_dir)
        if f.endswith(".jpg") or f.endswith(".png")
    ]

    if len(frame_files) == 0:
        raise RuntimeError("No frames found in directory")

    # Sort frames by the last number in the filename
    def extract_number(filename):
        nums = re.findall(r"\d+", filename)
        return int(nums[-1]) if nums else -1

    frame_files.sort(key=extract_number)

    # Read first frame to get size
    first_frame = cv2.imread(os.path.join(frames_dir, frame_files[0]))
    height, width = first_frame.shape[:2]

    # Output path inside the same folder
    output_path = os.path.join(frames_dir, video_name)

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Write frames
    for f in frame_files:
        frame_path = os.path.join(frames_dir, f)
        frame = cv2.imread(frame_path)
        video.write(frame)

    video.release()

    print(f"Video saved to: {output_path}")



frames_to_video("/home/bera/Pictures/Simulation Videos/CASE-1/Proposed/topdown_annotated", fps=5)