import cv2
import os
import numpy as np
from typing import Tuple, Optional

def overlay_circle_region(
    img_bgr: np.ndarray,
    *,
    center_xy: Tuple[int, int],
    radius_px: int,
    fill_bgr: Tuple[int, int, int] = (0, 255, 0),
    fill_alpha: float = 0.22,
    border_bgr: Tuple[int, int, int] = (0, 180, 0),
    border_thickness: int = 4,
    label: Optional[str] = None,
    label_scale: float = 1.5,
    label_thickness: int = 3,
) -> np.ndarray:
    """
    Draw a transparent filled circle + solid border (looks like your green goal circles).
    img_bgr: OpenCV image (H,W,3) uint8 in BGR.
    """
    out = img_bgr.copy()

    # Transparent fill
    overlay = out.copy()
    cv2.circle(overlay, center_xy, radius_px, fill_bgr, thickness=-1, lineType=cv2.LINE_AA)
    out = cv2.addWeighted(overlay, fill_alpha, out, 1.0 - fill_alpha, 0)

    # Border
    cv2.circle(out, center_xy, radius_px, border_bgr, thickness=border_thickness, lineType=cv2.LINE_AA)

    # Label
    if label is not None:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, label_scale, label_thickness)
        org = (center_xy[0] - tw // 2, center_xy[1] + th // 2)
        cv2.putText(out, label, org, cv2.FONT_HERSHEY_SIMPLEX, label_scale, (0, 0, 0),
                    label_thickness, cv2.LINE_AA)

    return out


def overlay_square_region(
    img_bgr: np.ndarray,
    *,
    top_left_xy: Tuple[int, int],
    bottom_right_xy: Tuple[int, int],
    fill_bgr: Tuple[int, int, int] = (0, 0, 255),
    fill_alpha: float = 0.3,
    border_bgr: Tuple[int, int, int] = (40, 40, 40),
    border_thickness: int = 1,
    text: Optional[str] = "Forbidden\nRegion",
    text_scale: float = 1,
    text_thickness: int = 2,
    text_color_bgr: Tuple[int, int, int] = (255, 255, 255),
    line_spacing: int = 18,
) -> np.ndarray:
    """
    Draw a transparent filled square/rect + solid border + centered multi-line text
    (looks like your red forbidden region box).
    """
    out = img_bgr.copy()
    x1, y1 = top_left_xy
    x2, y2 = bottom_right_xy

    # Transparent fill
    overlay = out.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), fill_bgr, thickness=-1, lineType=cv2.LINE_AA)
    out = cv2.addWeighted(overlay, fill_alpha, out, 1.0 - fill_alpha, 0)

    # Border
    cv2.rectangle(out, (x1, y1), (x2, y2), border_bgr, thickness=border_thickness, lineType=cv2.LINE_AA)

    # Centered multi-line text
    if text is not None:
        lines = text.split("\n")
        sizes = [cv2.getTextSize(ln, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_thickness)[0] for ln in lines]
        total_h = sum(h for (_, h) in sizes) + line_spacing * (len(lines) - 1)

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        y = cy - total_h // 2
        for (ln, (tw, th)) in zip(lines, sizes):
            x = cx - tw // 2
            y = y + th
            cv2.putText(out, ln, (x, y), cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_color_bgr,
                        text_thickness, cv2.LINE_AA)
            y = y + line_spacing

    return out


def overlay_regions_on_topdown(
    img_bgr: np.ndarray,
    *,
    green_circles: Optional[list] = None,
    red_squares: Optional[list] = None,
) -> np.ndarray:
    """
    Convenience wrapper to apply multiple regions.

    green_circles item format:
      dict(center_xy=(x,y), radius_px=r, label="1")  # plus any overlay_circle_region kwargs

    red_squares item format:
      dict(top_left_xy=(x1,y1), bottom_right_xy=(x2,y2), text="Forbidden\nRegion")  # plus kwargs
    """
    out = img_bgr.copy()

    if green_circles:
        for c in green_circles:
            out = overlay_circle_region(out, **c)

    if red_squares:
        for s in red_squares:
            out = overlay_square_region(out, **s)

    return out

def annotate_frames_in_folder(
    input_dir,
    output_dir,
    green_circles=None,
    red_squares=None
):
    """
    Applies overlay regions to all images in a folder.

    Parameters
    ----------
    input_dir : str
        Folder containing frames
    output_dir : str
        Folder to save annotated frames
    green_circles : list of dict
        Arguments for overlay_circle_region
    red_squares : list of dict
        Arguments for overlay_square_region
    """

    os.makedirs(output_dir, exist_ok=True)

    valid_ext = (".jpg", ".jpeg", ".png")

    files = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith(valid_ext)
    ])

    for file in files:

        img_path = os.path.join(input_dir, file)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Skipping unreadable file: {file}")
            continue

        # Apply circle regions
        if green_circles:
            for circle in green_circles:
                img = overlay_circle_region(img, **circle)

        # Apply square regions
        if red_squares:
            for square in red_squares:
                img = overlay_square_region(img, **square)

        # Build new filename
        name, ext = os.path.splitext(file)
        out_name = f"{name}_annotated{ext}"
        out_path = os.path.join(output_dir, out_name)

        # Save image
        cv2.imwrite(out_path, img)

    print(f"Annotated frames saved to: {output_dir}")


if __name__ == "__main__":
    #CASE-2:
#     annotate_frames_in_folder(
#     input_dir="/home/bera/Pictures/Simulation Videos/CASE-2/SPOC Path/topdown",
#     output_dir="/home/bera/Pictures/Simulation Videos/CASE-2/SPOC Path/topdown_annotated",

#     green_circles=[
#             dict(center_xy=(160, 310), radius_px=60, label="1"),
#             dict(center_xy=(690, 370), radius_px=60, label="2"),
#             dict(center_xy=(520, 750), radius_px=60, label="3"),
#     ],

#     red_squares=[
#         dict(top_left_xy=(300, 320), bottom_right_xy=(515, 535), text="Forbidden\nRegion")
#     ]
# )

    #CASE-1:
    annotate_frames_in_folder(
    input_dir="/home/bera/Pictures/Simulation Videos/CASE-1/Proposed/topdown",
    output_dir="/home/bera/Pictures/Simulation Videos/CASE-1/Proposed/topdown_annotated",

    green_circles=[
            dict(center_xy=(895, 125), radius_px=50, label="1"),
            dict(center_xy=(150, 600), radius_px=50, label="2"),
    ])


