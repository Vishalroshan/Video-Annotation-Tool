import os
import sys
import numpy as np
import torch
import cv2
from PIL import Image

from sam2.build_sam import build_sam2_video_predictor

# Device setup
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# Fix for dtype mismatch error when using multiple masks
if device.type == "cuda":
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "cpu":
    torch.autocast("cpu", dtype=torch.bfloat16).__enter__()

# Annotation classes
OBJ_NAMES = [
    "Patient", "Anest", "Assistant Surgeon", "Circulator", "Head Surgeon",
    "MPS", "Nurse", "Student", "Unrelated Person", "Instrument",
    "AE", "C-Arm", "Drill", "Hammer", "Instrument Table",
    "Tracker", "Mako Robot", "Monitor", "MPS Station", "OT",
    "Saw", "Secondary Table", "Cementer", "Drape"
]

# Colors in BGR format for OpenCV
OBJ_COLORS = [
    (0, 255, 255),      # patient
    (0, 26, 135),       # anest
    (0, 194, 255),      # assistant_surgeon
    (0, 135, 135),      # circulator
    (0, 74, 26),        # head_surgeon
    (0, 255, 26),       # mps
    (0, 242, 145),      # nurse
    (0, 145, 242),      # student
    (0, 74, 74),        # unrelated_person
    (0, 87, 255),       # instrument
    (0, 232, 207),      # ae
    (0, 74, 194),       # c_arm
    (0, 184, 122),      # drill
    (0, 122, 184),      # hammer
    (0, 26, 255),       # instrument_table
    (0, 87, 122),       # tracker
    (0, 184, 184),      # mako_robot
    (0, 255, 87),       # monitor
    (0, 26, 26),        # mps_station
    (0, 194, 74),       # ot
    (0, 122, 87),       # saw
    (0, 184, 26),       # secondary_table
    (0, 135, 26),       # cementer
    (0, 26, 184),       # drape
]

MAX_OBJECTS = len(OBJ_NAMES)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output_video_masks")
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 100  # Number of frames per batch

# Get total frame count from video or folder


def get_total_frame_count(path):
    if os.path.isdir(path):
        jpg_files = sorted([
            f for f in os.listdir(path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])
        return len(jpg_files)
    elif os.path.isfile(path):
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return total
    else:
        raise FileNotFoundError(f"{path} does not exist!")

# Load frames from folder


def load_frames_from_folder(folder_path, start_idx, count):
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"{folder_path} does not exist!")

    jpg_files = sorted([
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    end_idx = min(start_idx + count, len(jpg_files))
    jpg_files_subset = jpg_files[start_idx:end_idx]

    frames = [np.array(Image.open(f)) for f in jpg_files_subset]
    print(f"Loaded {len(frames)} frames (indices {start_idx}-{end_idx-1})")
    return frames

# Load video frames


def load_video_frames(video_path, start_idx, count):
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"{video_path} does not exist!")

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)

    frames = []
    for _ in range(count):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()

    print(
        f"Loaded {len(frames)} frames (indices {start_idx}-{start_idx+len(frames)-1})")
    return frames

# Load either video or folder


def load_frames(path, start_idx, count):
    if os.path.isdir(path):
        return load_frames_from_folder(path, start_idx, count)
    elif os.path.isfile(path):
        return load_video_frames(path, start_idx, count)
    else:
        raise FileNotFoundError(f"{path} does not exist!")


# Global state
objects_data = {
    obj_id: {"points": {}, "labels": {}, "masks": {}}
    for obj_id in range(MAX_OBJECTS)
}
active_obj = 0
cur_frame_idx = 0
frames = []
video_segments = {}
last_removed_point = None

# Batch state
current_batch = 0
total_batches = 0
batch_start_idx = 0
video_path_global = None
total_frame_count = 0

# Previous batch's last frame masks for propagation
previous_batch_masks = {}

# SAM2 state (reset each batch)
sam2_inference_state = None
sam2_predictor = None

# Window names
MAIN_WINDOW = "Video Annotation Tool"
SELECTOR_WINDOW = "Object Selection"

# Drawing functions


def create_selector_panel():
    panel_width = 400
    button_height = 50
    header_height = 80
    footer_height = 200
    buttons_area_height = button_height * len(OBJ_NAMES)
    panel_height = header_height + buttons_area_height + footer_height

    panel = np.ones((panel_height, panel_width, 3), dtype=np.uint8) * 45

    # HEADER SECTION
    cv2.rectangle(panel, (0, 0), (panel_width,
                  header_height), (60, 60, 60), -1)
    cv2.putText(panel, "Object Selection", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

    batch_text = f"Batch {current_batch+1}/{total_batches}"
    cv2.putText(panel, batch_text, (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

    # BUTTONS SECTION
    y_offset = header_height

    for i, obj_name in enumerate(OBJ_NAMES):
        y_start = y_offset + i * button_height
        y_end = y_offset + (i + 1) * button_height

        if i == active_obj:
            color = OBJ_COLORS[i]
            for j in range(button_height):
                alpha = 0.7 + 0.3 * (j / button_height)
                row_color = tuple(int(c * alpha) for c in color)
                cv2.line(panel, (5, y_start + j),
                         (panel_width - 5, y_start + j), row_color, 1)

            cv2.rectangle(panel, (5, y_start + 2),
                          (panel_width - 5, y_end - 2), (255, 255, 255), 3)
            text_color = (255, 255, 255)
            font_weight = 3
        else:
            cv2.rectangle(panel, (5, y_start + 2),
                          (panel_width - 5, y_end - 2), (70, 70, 70), -1)
            cv2.rectangle(panel, (5, y_start + 2),
                          (panel_width - 5, y_end - 2), (90, 90, 90), 1)
            text_color = (200, 200, 200)
            font_weight = 2

        color_box_size = 35
        margin = 15
        cv2.rectangle(panel,
                      (margin, y_start + 7),
                      (margin + color_box_size, y_end - 7),
                      OBJ_COLORS[i], -1)
        cv2.rectangle(panel,
                      (margin, y_start + 7),
                      (margin + color_box_size, y_end - 7),
                      (255, 255, 255) if i == active_obj else (150, 150, 150), 2)

        text_x = margin + color_box_size + 15
        text_y = y_start + 32
        cv2.putText(panel, obj_name, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, font_weight)

        total_points = sum(len(objects_data[i]["points"].get(
            f, [])) for f in range(len(frames)))
        if total_points > 0:
            count_text = f"{total_points}pts"
            cv2.putText(panel, count_text, (panel_width - 80, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 100), 2)

    # FOOTER SECTION
    footer_y = header_height + buttons_area_height
    cv2.rectangle(panel, (0, footer_y), (panel_width,
                  panel_height), (50, 50, 50), -1)

    controls_y = footer_y + 25
    cv2.putText(panel, "CONTROLS", (20, controls_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 3)

    controls = [
        ("Left Click", "Add positive point"),
        ("Middle Click", "Add negative point"),
        ("Shift+Click", "Remove point"),
        ("A / Z", "Switch object"),
        ("; / '", "Navigate frames"),
        ("[ / ]", "Jump 5 frames"),
        ("R", "SAM2 current frame"),
        ("M", "Merge & propagate"),
        ("T", "Save masks")
    ]

    y = controls_y + 20
    for key, desc in controls:
        cv2.putText(panel, key, (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 200, 255), 2)
        cv2.putText(panel, desc, (120, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 2)
        y += 20

    return panel


def draw_frame():
    global cur_frame_idx

    if cur_frame_idx >= len(frames):
        return np.zeros((480, 640, 3), dtype=np.uint8)

    display = frames[cur_frame_idx].copy()

    # Draw masks
    for obj_id, data in objects_data.items():
        if cur_frame_idx in data["masks"]:
            mask = data["masks"][cur_frame_idx]
            if mask.ndim == 3 and mask.shape[0] == 1:
                mask = mask[0]

            color_bgr = OBJ_COLORS[obj_id]
            color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])

            color_mask = np.zeros_like(display)
            color_mask[mask > 0] = color_rgb

            display = cv2.addWeighted(display, 1.0, color_mask, 0.5, 0)

            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(
                mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(display, contours, -1, color_rgb, 2)

    # Draw points
    for obj_id, data in objects_data.items():
        color_bgr = OBJ_COLORS[obj_id]
        color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
        points = data["points"].get(cur_frame_idx, [])
        labels = data["labels"].get(cur_frame_idx, [])

        for point, label in zip(points, labels):
            x, y = int(point[0]), int(point[1])
            if label == 1:  # Positive point
                cv2.circle(display, (x, y), 10, (255, 255, 255), -1)
                cv2.circle(display, (x, y), 8, color_rgb, -1)
                cv2.circle(display, (x, y), 10, (0, 0, 0), 2)
            else:  # Negative point
                cv2.circle(display, (x, y), 10, color_rgb, -1)
                cv2.circle(display, (x, y), 10, (255, 255, 255), 2)

    # Highlight removed point
    if last_removed_point is not None:
        x, y = int(last_removed_point[0]), int(last_removed_point[1])
        for radius in range(15, 25, 2):
            cv2.circle(display, (x, y), radius, (255, 255, 0), 2)

    return display

# Mouse callbacks


def mouse_callback(event, x, y, flags, param):
    global objects_data, active_obj, cur_frame_idx, last_removed_point

    # Get the actual frame dimensions (info bar is below the frame)
    if cur_frame_idx >= len(frames):
        return

    frame_height = frames[cur_frame_idx].shape[0]

    # Check if click is in info bar area (for buttons)
    if y >= frame_height:
        if event == cv2.EVENT_LBUTTONDOWN:
            handle_button_click(x, y - frame_height)
        return

    # Handle frame area clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        if flags & cv2.EVENT_FLAG_SHIFTKEY:
            # Shift + left click = remove nearest point
            if cur_frame_idx in objects_data[active_obj]["points"]:
                pts = np.array(objects_data[active_obj]
                               ["points"][cur_frame_idx])
                if len(pts) > 0:
                    dists = np.linalg.norm(pts - np.array([x, y]), axis=1)
                    idx = np.argmin(dists)
                    last_removed_point = objects_data[active_obj]["points"][cur_frame_idx].pop(
                        idx)
                    objects_data[active_obj]["labels"][cur_frame_idx].pop(idx)
        else:
            # Regular left click = add positive point
            if cur_frame_idx not in objects_data[active_obj]["points"]:
                objects_data[active_obj]["points"][cur_frame_idx] = []
                objects_data[active_obj]["labels"][cur_frame_idx] = []
            objects_data[active_obj]["points"][cur_frame_idx].append([x, y])
            objects_data[active_obj]["labels"][cur_frame_idx].append(1)
            last_removed_point = None

    elif event == cv2.EVENT_MBUTTONDOWN:
        # Middle mouse button = add negative point
        if cur_frame_idx not in objects_data[active_obj]["points"]:
            objects_data[active_obj]["points"][cur_frame_idx] = []
            objects_data[active_obj]["labels"][cur_frame_idx] = []
        objects_data[active_obj]["points"][cur_frame_idx].append([x, y])
        objects_data[active_obj]["labels"][cur_frame_idx].append(0)
        last_removed_point = None


def handle_button_click(x, y):
    global cur_frame_idx

    button_y = 85
    button_height = 30

    # Check if click is within button y range
    if y < button_y or y > button_y + button_height:
        return

    # Define button regions
    buttons = [
        ("<<5", 10, 60),
        ("<1", 75, 50),
        ("1>", 130, 50),
        ("5>>", 185, 60),
        ("Single", 260, 70),
        ("Batch", 335, 70),
    ]

    for label, x_start, btn_width in buttons:
        if x_start <= x <= x_start + btn_width:
            if label == "<<5":
                cur_frame_idx = max(cur_frame_idx - 5, 0)
                cv2.setTrackbarPos("Frame", MAIN_WINDOW, cur_frame_idx)
            elif label == "<1":
                cur_frame_idx = max(cur_frame_idx - 1, 0)
                cv2.setTrackbarPos("Frame", MAIN_WINDOW, cur_frame_idx)
            elif label == "1>":
                cur_frame_idx = min(cur_frame_idx + 1, len(frames) - 1)
                cv2.setTrackbarPos("Frame", MAIN_WINDOW, cur_frame_idx)
            elif label == "5>>":
                cur_frame_idx = min(cur_frame_idx + 5, len(frames) - 1)
                cv2.setTrackbarPos("Frame", MAIN_WINDOW, cur_frame_idx)
            elif label == "Single":
                run_prediction_single_frame()
            elif label == "Batch":
                merge_and_propagate()
            break


def selector_mouse_callback(event, x, y, flags, param):
    global active_obj

    if event == cv2.EVENT_LBUTTONDOWN:
        header_height = 80
        button_height = 50

        if y >= header_height:
            clicked_obj = (y - header_height) // button_height
            if 0 <= clicked_obj < len(OBJ_NAMES):
                active_obj = clicked_obj

# Trackbar callback


def on_trackbar(val):
    global cur_frame_idx, last_removed_point
    cur_frame_idx = val
    last_removed_point = None

# Batch management


def load_batch(batch_idx, save_masks_for_next=False):
    global frames, cur_frame_idx, batch_start_idx, objects_data, video_segments
    global last_removed_point, sam2_inference_state, previous_batch_masks

    # Save current batch's last frame masks if requested
    if save_masks_for_next and len(frames) > 0:
        last_frame_idx = len(frames) - 1
        if last_frame_idx in video_segments:
            previous_batch_masks = {
                obj_id: mask.copy() for obj_id, mask in video_segments[last_frame_idx].items()}
            print(
                f"Saved {len(previous_batch_masks)} masks from frame {batch_start_idx + last_frame_idx} for propagation")
        else:
            print("No masks found on last frame to save for propagation")

    batch_start_idx = batch_idx * BATCH_SIZE
    frames_to_load = min(BATCH_SIZE, total_frame_count - batch_start_idx)

    frames[:] = load_frames(video_path_global, batch_start_idx, frames_to_load)

    # Reset state for new batch
    objects_data = {
        obj_id: {"points": {}, "labels": {}, "masks": {}}
        for obj_id in range(MAX_OBJECTS)
    }
    video_segments.clear()
    last_removed_point = None
    sam2_inference_state = None  # Force reinitialization

    # Copy previous batch masks to frame 0 for visualization
    if len(previous_batch_masks) > 0:
        print(
            f"Copying {len(previous_batch_masks)} masks from previous batch to frame 0 for visualization")
        video_segments[0] = {}
        for obj_id, mask in previous_batch_masks.items():
            mask_copy = mask.copy()
            video_segments[0][obj_id] = mask_copy
            objects_data[obj_id-1]["masks"][0] = mask_copy.astype(np.uint8)
        print(f"Masks copied to frame 0. You can now see where objects were before running SAM2.")

    print(
        f"Loaded batch {batch_idx + 1}/{total_batches} (frames {batch_start_idx}-{batch_start_idx+len(frames)-1})")

    if len(previous_batch_masks) > 0:
        print(
            f"Previous batch has {len(previous_batch_masks)} masks available for SAM2 initialization")

    cur_frame_idx = 0
    try:
        cv2.setTrackbarPos("Frame", MAIN_WINDOW, 0)
        cv2.setTrackbarMax("Frame", MAIN_WINDOW, len(frames) - 1)
    except:
        pass


def next_batch():
    global current_batch

    if current_batch < total_batches - 1:
        # Auto-save masks from current batch before moving
        if len(video_segments) > 0:
            print(f"\nAuto-saving masks from batch {current_batch + 1}...")
            auto_save_masks()

        # Save masks before changing batch
        load_batch(current_batch + 1, save_masks_for_next=True)
        current_batch += 1
    else:
        print("Already at last batch!")


def prev_batch():
    global current_batch, previous_batch_masks

    if current_batch > 0:
        previous_batch_masks = {}  # Don't propagate when going backwards
        current_batch -= 1
        load_batch(current_batch, save_masks_for_next=False)
    else:
        print("Already at first batch!")

# Run SAM2 (CURRENT FRAME ONLY)


def run_prediction_single_frame():
    global video_segments, objects_data

    has_points = any(len(data["points"].get(
        cur_frame_idx, [])) > 0 for data in objects_data.values())

    if not has_points:
        print(
            f"No points on current frame {cur_frame_idx}. Please add points first.")
        return

    print(f"Running SAM2 on frame {batch_start_idx + cur_frame_idx} only...")

    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        # Use image predictor for single frame
        sam2_model = build_sam2(
            "configs/sam2.1/sam2.1_hiera_l.yaml",
            "../checkpoints/sam2.1_hiera_large.pt",
            device=device,
        )
        predictor = SAM2ImagePredictor(sam2_model)

        # Set current frame
        predictor.set_image(frames[cur_frame_idx])

        # Process each object with points on current frame
        for obj_id, data in objects_data.items():
            if cur_frame_idx in data["points"] and len(data["points"][cur_frame_idx]) > 0:
                points = np.array(
                    data["points"][cur_frame_idx], dtype=np.float32)
                labels = np.array(
                    data["labels"][cur_frame_idx], dtype=np.int32)

                # Predict mask for this object
                masks, scores, logits = predictor.predict(
                    point_coords=points,
                    point_labels=labels,
                    multimask_output=False,
                )

                # Store mask
                mask = masks[0]  # Take first mask
                if cur_frame_idx not in video_segments:
                    video_segments[cur_frame_idx] = {}
                video_segments[cur_frame_idx][obj_id+1] = mask
                objects_data[obj_id]["masks"][cur_frame_idx] = mask.astype(
                    np.uint8)

                print(
                    f"Generated mask for {OBJ_NAMES[obj_id]} on frame {batch_start_idx + cur_frame_idx}")

        print(f"Single frame prediction done!")

    except Exception as e:
        print(f"ERROR during single frame prediction: {e}")
        import traceback
        traceback.print_exc()

# Merge and propagate all masks through batch FROM CURRENT FRAME FORWARD


def merge_and_propagate():
    global video_segments, sam2_inference_state, sam2_predictor

    # Collect all frames that have masks
    frames_with_masks = sorted(video_segments.keys())

    if len(frames_with_masks) == 0:
        print("No masks to propagate. Use 'R' to generate masks first.")
        return

    # Determine starting frame for propagation
    start_frame = cur_frame_idx

    print(
        f"Merging and propagating from frame {batch_start_idx + start_frame} forward through remaining frames...")

    try:
        # Clear CUDA cache
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("Initializing SAM2 predictor...")
        sam2_predictor = build_sam2_video_predictor(
            "configs/sam2.1/sam2.1_hiera_l.yaml",
            "../checkpoints/sam2.1_hiera_large.pt",
            device=device,
        )

        # Save frames to disk
        frames_dir = "./frames_from_video"
        if os.path.exists(frames_dir):
            import shutil
            shutil.rmtree(frames_dir)
        os.makedirs(frames_dir, exist_ok=True)

        print(f"Saving {len(frames)} frames to disk...")
        for i, frame in enumerate(frames):
            Image.fromarray(frame).save(
                os.path.join(frames_dir, f"{i:05d}.jpg"))

        # Initialize inference state
        sam2_inference_state = sam2_predictor.init_state(video_path=frames_dir)
        print(f"SAM2 initialized with {len(frames)} frames")

        # Add masks from current frame or earlier annotated frames
        objects_added = set()

        # Find the most recent frame with masks at or before current frame
        relevant_frames = [f for f in frames_with_masks if f <= start_frame]
        if len(relevant_frames) == 0:
            print(f"No masks found at or before frame {start_frame}!")
            return

        # Use the most recent annotated frame as starting point
        init_frame = relevant_frames[-1]

        for obj_id, mask in video_segments[init_frame].items():
            mask_to_add = np.squeeze(mask)

            try:
                sam2_predictor.add_new_mask(
                    inference_state=sam2_inference_state,
                    frame_idx=init_frame,
                    obj_id=obj_id,
                    mask=mask_to_add
                )
                objects_added.add(obj_id)
                print(
                    f"Added mask for {OBJ_NAMES[obj_id-1]} at frame {batch_start_idx + init_frame}")
            except Exception as e:
                print(
                    f"Error adding mask for {OBJ_NAMES[obj_id-1]} at frame {init_frame}: {e}")

        if len(objects_added) == 0:
            print("No masks could be added for propagation!")
            return

        # Preserve masks before start_frame
        preserved_masks = {f: segs.copy()
                           for f, segs in video_segments.items() if f < start_frame}

        # Clear masks from start_frame onwards
        for f in list(video_segments.keys()):
            if f >= start_frame:
                del video_segments[f]

        print(
            f"Propagating from frame {batch_start_idx + start_frame} through frame {batch_start_idx + len(frames) - 1}...")

        # Propagate from start_frame to end
        for out_frame_idx, out_obj_ids, out_mask_logits in sam2_predictor.propagate_in_video(
            inference_state=sam2_inference_state,
            start_frame_idx=start_frame
        ):
            if out_frame_idx not in video_segments:
                video_segments[out_frame_idx] = {}

            for i, out_obj_id in enumerate(out_obj_ids):
                mask = (out_mask_logits[i] > 0.0).cpu().numpy()
                video_segments[out_frame_idx][out_obj_id] = mask
                objects_data[out_obj_id -
                             1]["masks"][out_frame_idx] = mask.astype(np.uint8)

        # Restore preserved masks
        for f, segs in preserved_masks.items():
            video_segments[f] = segs
            for obj_id, mask in segs.items():
                objects_data[obj_id-1]["masks"][f] = mask.astype(np.uint8)

        print(
            f"Propagation done! Processed frames {batch_start_idx + start_frame} to {batch_start_idx + len(frames) - 1}")

    except Exception as e:
        print(f"ERROR during merge and propagate: {e}")
        import traceback
        traceback.print_exc()

# Run SAM2 prediction (OLD - kept for first batch compatibility)


def run_prediction():
    global video_segments, sam2_inference_state, sam2_predictor

    has_points = any(len(data["points"]) > 0 for data in objects_data.values())
    has_previous_masks = len(previous_batch_masks) > 0 and current_batch > 0

    if not has_points and not has_previous_masks:
        print("No points or previous masks available. Please annotate some objects first.")
        return

    print("Running SAM2 predictions...")

    # CRITICAL: Always reinitialize BOTH predictor and inference state to avoid dtype conflicts
    print("Reinitializing SAM2 predictor to avoid dtype conflicts...")
    sam2_inference_state = None

    try:
        # ALWAYS reinitialize the predictor to clear internal dtype state
        print("Initializing SAM2 predictor...")

        # Clear CUDA cache to help with dtype issues
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        sam2_predictor = build_sam2_video_predictor(
            "configs/sam2.1/sam2.1_hiera_l.yaml",
            "../checkpoints/sam2.1_hiera_large.pt",
            device=device,
        )

        # Save current batch frames to disk
        frames_dir = "./frames_from_video"
        if os.path.exists(frames_dir):
            import shutil
            shutil.rmtree(frames_dir)
        os.makedirs(frames_dir, exist_ok=True)

        print(f"Saving {len(frames)} frames to disk...")
        for i, frame in enumerate(frames):
            Image.fromarray(frame).save(
                os.path.join(frames_dir, f"{i:05d}.jpg"))

        # Initialize inference state (FRESH each time to avoid dtype issues)
        sam2_inference_state = sam2_predictor.init_state(video_path=frames_dir)
        print(f"SAM2 initialized with {len(frames)} frames")

        # Initialize objects_to_propagate set
        objects_to_propagate = set()

        # Use previous batch's masks as initialization for frame 0
        if has_previous_masks and cur_frame_idx == 0:
            print(
                f"Initializing frame 0 with {len(previous_batch_masks)} masks from previous batch")
            for obj_id, mask in previous_batch_masks.items():
                try:
                    # Ensure mask is in correct format
                    mask_to_add = np.squeeze(mask)

                    # Check mask properties for debugging
                    print(
                        f"  Mask for {OBJ_NAMES[obj_id-1]}: shape={mask_to_add.shape}, dtype={mask_to_add.dtype}, min={mask_to_add.min()}, max={mask_to_add.max()}")

                    # Use add_new_mask method
                    sam2_predictor.add_new_mask(
                        inference_state=sam2_inference_state,
                        frame_idx=0,
                        obj_id=obj_id,
                        mask=mask_to_add
                    )
                    print(
                        f"  ✓ Added mask for {OBJ_NAMES[obj_id-1]} at frame 0")
                    objects_to_propagate.add(obj_id)
                except Exception as e:
                    print(
                        f"  ✗ Error adding mask for {OBJ_NAMES[obj_id-1]}: {e}")
                    print(f"  Falling back to bounding box...")
                    # Fallback: extract bounding box from mask
                    mask_2d = np.squeeze(mask)
                    y_indices, x_indices = np.where(mask_2d > 0)
                    if len(x_indices) > 0 and len(y_indices) > 0:
                        x_min, x_max = x_indices.min(), x_indices.max()
                        y_min, y_max = y_indices.min(), y_indices.max()
                        box = np.array(
                            [[x_min, y_min, x_max, y_max]], dtype=np.float32)

                        sam2_predictor.add_new_points_or_box(
                            inference_state=sam2_inference_state,
                            frame_idx=0,
                            obj_id=obj_id,
                            box=box
                        )
                        print(
                            f"  ✓ Added bounding box for {OBJ_NAMES[obj_id-1]} at frame 0")
                        objects_to_propagate.add(obj_id)

        # Add user-provided points
        for obj_id, data in objects_data.items():
            if cur_frame_idx in data["points"] and len(data["points"][cur_frame_idx]) > 0:
                points = np.array(
                    data["points"][cur_frame_idx], dtype=np.float32)
                labels = np.array(
                    data["labels"][cur_frame_idx], dtype=np.int32)

                sam2_predictor.add_new_points_or_box(
                    inference_state=sam2_inference_state,
                    frame_idx=cur_frame_idx,
                    obj_id=obj_id+1,
                    points=points,
                    labels=labels
                )
                objects_to_propagate.add(obj_id+1)
                print(
                    f"Added {len(points)} points for {OBJ_NAMES[obj_id]} at frame {batch_start_idx + cur_frame_idx}")

        if not objects_to_propagate and not has_previous_masks:
            print("No new points or masks to propagate!")
            return

        # Clear existing masks
        video_segments.clear()
        for obj_id in range(MAX_OBJECTS):
            objects_data[obj_id]["masks"].clear()

        print(f"Propagating through all {len(frames)} frames...")

        # Propagate through entire batch
        for out_frame_idx, out_obj_ids, out_mask_logits in sam2_predictor.propagate_in_video(
            inference_state=sam2_inference_state,
            start_frame_idx=0
        ):
            if out_frame_idx not in video_segments:
                video_segments[out_frame_idx] = {}

            for i, out_obj_id in enumerate(out_obj_ids):
                mask = (out_mask_logits[i] > 0.0).cpu().numpy()
                video_segments[out_frame_idx][out_obj_id] = mask
                objects_data[out_obj_id -
                             1]["masks"][out_frame_idx] = mask.astype(np.uint8)

        print(f"Prediction done! Processed {len(video_segments)} frames")

    except Exception as e:
        print(f"ERROR during SAM2 prediction: {e}")
        import traceback
        traceback.print_exc()

# Auto-save masks (called when moving to next batch)


def auto_save_masks():
    output_dir = os.path.join(BASE_DIR, "output_video_masks")
    os.makedirs(output_dir, exist_ok=True)

    saved_count = 0
    for fidx, segs in video_segments.items():
        global_frame_idx = batch_start_idx + fidx
        h, w, _ = frames[fidx].shape
        combined = np.zeros((h, w, 3), dtype=np.uint8)

        for obj_id, mask in segs.items():
            mask_img = np.squeeze(mask)
            mask_img = (mask_img * 255).astype(np.uint8)

            color_bgr = np.array(OBJ_COLORS[obj_id-1])
            color_rgb = np.array([color_bgr[2], color_bgr[1], color_bgr[0]])

            for c in range(3):
                combined[..., c] += (mask_img * color_rgb[c] //
                                     255).astype(np.uint8)

        combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(
            output_dir, f"frame_{global_frame_idx:05d}_combined.png"), combined_bgr)
        saved_count += 1

    print(f"✓ Auto-saved {saved_count} masks to {output_dir}")

# Save masks


def save_masks():
    folder_name = input(
        "Enter folder name to save masks (default: output_video_masks): ").strip()
    if folder_name == "":
        folder_name = "output_video_masks"

    output_dir = os.path.join(BASE_DIR, folder_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Saving {len(video_segments)} masks...")

    for fidx, segs in video_segments.items():
        global_frame_idx = batch_start_idx + fidx
        h, w, _ = frames[fidx].shape
        combined = np.zeros((h, w, 3), dtype=np.uint8)

        for obj_id, mask in segs.items():
            mask_img = np.squeeze(mask)
            mask_img = (mask_img * 255).astype(np.uint8)

            color_bgr = np.array(OBJ_COLORS[obj_id-1])
            color_rgb = np.array([color_bgr[2], color_bgr[1], color_bgr[0]])

            for c in range(3):
                combined[..., c] += (mask_img * color_rgb[c] //
                                     255).astype(np.uint8)

        combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(
            output_dir, f"frame_{global_frame_idx:05d}_combined.png"), combined_bgr)

    print(f"✓ Saved {len(video_segments)} masks to {output_dir}")
    cv2.waitKey(1)


def main(input_path):
    global frames, total_batches, current_batch, batch_start_idx
    global active_obj, cur_frame_idx, last_removed_point
    global video_path_global, total_frame_count

    video_path_global = input_path
    total_frame_count = get_total_frame_count(input_path)
    total_batches = (total_frame_count + BATCH_SIZE - 1) // BATCH_SIZE
    current_batch = 0

    print(f"Total frames: {total_frame_count}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Total batches: {total_batches}")

    print("\n=== Controls ===")
    print("Left click: Add positive point")
    print("Middle click: Add negative point")
    print("Shift+click: Remove nearest point")
    print("A/Z: Switch objects")
    print("; / ': Navigate frames (back/forward)")
    print("[ / ]: Jump 5 frames")
    print("R: Run SAM2 on CURRENT FRAME ONLY")
    print("M: Merge & Propagate all masks through batch")
    print("T: Save masks")
    print("D: Delete last point")
    print("N/P: Next/Prev batch")
    print("Q/ESC: Quit")
    print("================\n")

    cv2.namedWindow(MAIN_WINDOW, cv2.WINDOW_NORMAL)
    cv2.namedWindow(SELECTOR_WINDOW, cv2.WINDOW_NORMAL)

    cv2.resizeWindow(MAIN_WINDOW, 1280, 720)
    cv2.resizeWindow(SELECTOR_WINDOW, 400, 800)

    cv2.setMouseCallback(MAIN_WINDOW, mouse_callback)
    cv2.setMouseCallback(SELECTOR_WINDOW, selector_mouse_callback)

    # Load first batch
    load_batch(0, save_masks_for_next=False)

    # Create trackbar
    cv2.createTrackbar("Frame", MAIN_WINDOW, 0, len(frames)-1, on_trackbar)

    while True:
        display = draw_frame()

        # Add info panel
        h, w = display.shape[:2]
        info_bar_height = 120
        info_bar = np.zeros((info_bar_height, w, 3), dtype=np.uint8)
        info_bar[:] = (40, 40, 40)

        # Active object info
        active_color = OBJ_COLORS[active_obj]
        active_color_rgb = (active_color[2], active_color[1], active_color[0])
        cv2.rectangle(info_bar, (10, 10), (40, 40), active_color_rgb, -1)
        cv2.rectangle(info_bar, (10, 10), (40, 40), (255, 255, 255), 2)

        cv2.putText(info_bar, f"Active: {OBJ_NAMES[active_obj]}", (50, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # Frame info
        global_frame_idx = batch_start_idx + cur_frame_idx + 1
        frame_text = f"Frame: {global_frame_idx} / {total_frame_count}"
        cv2.putText(info_bar, frame_text, (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        # Batch info
        batch_text = f"Batch: {current_batch + 1} / {total_batches}"
        cv2.putText(info_bar, batch_text, (350, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        # Point count
        total_points = sum(len(objects_data[i]["points"].get(cur_frame_idx, []))
                           for i in range(MAX_OBJECTS))
        if total_points > 0:
            cv2.putText(info_bar, f"Points: {total_points}", (650, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)

        # Mask count
        mask_count = len([1 for data in objects_data.values()
                         if cur_frame_idx in data["masks"]])
        if mask_count > 0:
            cv2.putText(info_bar, f"Masks: {mask_count}", (850, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 255), 2)

        # Previous batch indicator
        if len(previous_batch_masks) > 0:
            cv2.putText(info_bar, f"Prev masks: {len(previous_batch_masks)}", (1050, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 100), 2)

        # NAVIGATION BUTTONS + SAM BUTTONS
        button_y = 85
        button_height = 30

        buttons = [
            ("<<5", 10, 60),
            ("<1", 75, 50),
            ("1>", 130, 50),
            ("5>>", 185, 60),
            ("Single", 260, 70),  # R button
            ("Batch", 335, 70),   # M button
        ]

        for i, (label, x_start, btn_width) in enumerate(buttons):
            # Different color for SAM buttons
            if label in ["Single", "Batch"]:
                bg_color = (100, 100, 150)
                border_color = (150, 150, 200)
            else:
                bg_color = (80, 80, 80)
                border_color = (150, 150, 150)

            cv2.rectangle(info_bar,
                          (x_start, button_y),
                          (x_start + btn_width, button_y + button_height),
                          bg_color, -1)
            cv2.rectangle(info_bar,
                          (x_start, button_y),
                          (x_start + btn_width, button_y + button_height),
                          border_color, 2)

            text_size = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            text_x = x_start + (btn_width - text_size[0]) // 2
            text_y = button_y + (button_height + text_size[1]) // 2
            cv2.putText(info_bar, label, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Combine frame and info bar
        combined = np.vstack([display, info_bar])

        cv2.imshow(MAIN_WINDOW, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

        selector = create_selector_panel()
        cv2.imshow(SELECTOR_WINDOW, selector)

        key = cv2.waitKey(10) & 0xFF

        # Only allow specific keys
        allowed_keys = [
            255,  # No key
            ord('q'), ord('Q'),
            ord('a'), ord('A'),
            ord('z'), ord('Z'),
            ord('r'), ord('R'),
            ord('m'), ord('M'),
            ord('t'), ord('T'),
            ord('n'), ord('N'),
            ord('p'), ord('P'),
            ord('d'), ord('D'),
            ord(';'),
            ord("'"),
            ord('['),
            ord(']'),
            27  # ESC
        ]

        if key not in allowed_keys:
            continue

        if key == 255:  # No key pressed
            continue
        elif key == ord('q') or key == ord('Q') or key == 27:  # Q or ESC
            break
        elif key == ord('a') or key == ord('A'):
            active_obj = (active_obj - 1) % MAX_OBJECTS
        elif key == ord('z') or key == ord('Z'):
            active_obj = (active_obj + 1) % MAX_OBJECTS
        elif key == ord('r') or key == ord('R'):
            run_prediction_single_frame()
        elif key == ord('m') or key == ord('M'):
            merge_and_propagate()
        elif key == ord('t') or key == ord('T'):
            save_masks()
        elif key == ord('n') or key == ord('N'):
            next_batch()
        elif key == ord('p') or key == ord('P'):
            prev_batch()
        elif key == ord('d') or key == ord('D'):
            if cur_frame_idx in objects_data[active_obj]["points"] and len(objects_data[active_obj]["points"][cur_frame_idx]) > 0:
                last_removed_point = objects_data[active_obj]["points"][cur_frame_idx].pop(
                )
                objects_data[active_obj]["labels"][cur_frame_idx].pop()
                print(f"Deleted last point for {OBJ_NAMES[active_obj]}")
        elif key == ord(';'):
            cur_frame_idx = max(cur_frame_idx - 1, 0)
            cv2.setTrackbarPos("Frame", MAIN_WINDOW, cur_frame_idx)
        elif key == ord("'"):
            cur_frame_idx = min(cur_frame_idx + 1, len(frames) - 1)
            cv2.setTrackbarPos("Frame", MAIN_WINDOW, cur_frame_idx)
        elif key == ord('['):
            cur_frame_idx = max(cur_frame_idx - 5, 0)
            cv2.setTrackbarPos("Frame", MAIN_WINDOW, cur_frame_idx)
        elif key == ord(']'):
            cur_frame_idx = min(cur_frame_idx + 5, len(frames) - 1)
            cv2.setTrackbarPos("Frame", MAIN_WINDOW, cur_frame_idx)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Disable Qt right-click menu if available
    import os
    os.environ["QT_QPA_PLATFORM"] = "xcb"

    if len(sys.argv) < 2:
        print("Usage: python3 vid_8.py <video.mp4 | frames_folder>")
        sys.exit(1)
    main(sys.argv[1])
