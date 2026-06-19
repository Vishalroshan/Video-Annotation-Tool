# SAM2 Video Annotation Tool

An interactive PyQt5-based tool for annotating surgical videos using Meta's Segment Anything Model 2 (SAM2). Videos are processed in 100-frame batches with automatic mask propagation across batch boundaries.

## Features

- **PyQt5 GUI**: Native windowed interface with dockable object panel, scrub slider, and live terminal log
- **Batch Processing**: 100-frame chunks to manage GPU memory efficiently
- **Interactive Annotation**: Click-based point annotation for object segmentation
- **Smart Propagation**: Automatically carries masks from the last frame of one batch into the first frame of the next
- **Frame-by-Frame Control**: Generate masks on a single frame (`R`) or propagate through the rest of the batch (`M`)
- **Auto-Save**: Masks are saved automatically when advancing to the next batch
- **Play/Pause**: Preview video with the Play button or Space key
- **5 Pre-defined Objects**: Head Surgeon, Assistant Surgeon, Nurse, Tool 1, Tool 2

## Requirements

### Hardware
- **GPU**: NVIDIA GPU with 24 GB VRAM (tested on Quadro RTX 6000)
- **RAM**: 64 GB+ recommended
- **Storage**: Sufficient space for video and output masks

### Software
- Python 3.10+
- CUDA 11.4+
- SAM2 installed from [facebookresearch/segment-anything-2](https://github.com/facebookresearch/segment-anything-2)

### Python Dependencies
```bash
pip install torch torchvision opencv-python pillow numpy PyQt5
```

## Installation

1. **Install SAM2**:
```bash
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2
pip install -e .
```

2. **Download SAM2 Checkpoint**:
```bash
cd checkpoints
./download_ckpts.sh
```

3. **Place the annotation script** anywhere — it searches for SAM2 checkpoints relative to its own location and common parent directories. Set `SAM2_ROOT` if it cannot be found automatically:
```bash
export SAM2_ROOT=/path/to/segment-anything-2
```

## Usage

### Launch

Run without arguments and open a file from the menu:
```bash
python annotation_tool.py
```

Or pass a video / folder directly to open it on launch:
```bash
python annotation_tool.py <video.mp4 | frames_folder>
```

### Opening Files

Use **File → Open Video…** (`Ctrl+O`) or **File → Open Folder of Frames…** (`Ctrl+Shift+O`).  
Supported video formats: `.mp4`, `.avi`, `.mkv`, `.mov`, `.webm`.  
Supported image formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`.

### UI Layout

```
┌─────────────────────────────────────────────────────────┐
│  Object Panel (dock)  │  Video display                  │
│  ─────────────────    │                                 │
│  Object Selection      │                                 │
│  Batch info            │                                 │
│  Object list           │                                 │
│  Controls reference    │                                 │
│  Terminal log box      │                                 │
│                        ├─────────────────────────────────│
│                        │  [< Prev] [Play] [Next >]       │
│                        │  Step: [1]  ──slider──  [label] │
│                        ├─────────────────────────────────│
│                        │  [R:Run] [M:Prop] [T:Save]      │
│                        │  [P:Prev Batch] [N:Next Batch]  │
└─────────────────────────────────────────────────────────┘
```

- **Object Panel**: Select the active object, shows point counts and batch info. Terminal output is mirrored to the log box at the bottom.
- **Controls Bar**: Frame scrubber, step size, play/pause, go-to field.
- **SAM Toolbar**: SAM2 action buttons and live status message.

### Annotation Workflow

#### Batch 1 (Initial Annotation)
1. **Select object** in the left panel (or press `A`/`Z`)
2. **Add points**: left-click to include, middle-click to exclude
3. **Generate mask**: press `R` — runs SAM2 on the current frame only
4. **Propagate**: press `M` — propagates masks from the current frame forward through the batch
5. **Advance**: press `N` — auto-saves masks and loads the next 100 frames

#### Batch 2+ (With Previous Masks)
1. Previous batch's last-frame masks appear on frame 0 automatically
2. **Option A – masks look good**: press `M` immediately to propagate
3. **Option B – corrections needed**:
   - Navigate to the problem frame
   - Add correction points and press `R`
   - Press `M` to re-propagate from that frame forward (frames before it are preserved)
4. Press `N` to advance

## Controls

### Mouse
| Action | Result |
|--------|--------|
| Left click | Add positive point (include in mask) |
| Middle click | Add negative point (exclude from mask) |
| Shift + left click | Remove nearest point |

### Keyboard
| Key | Action |
|-----|--------|
| `A` / `Z` | Previous / next object |
| `; ` / `'` | Navigate backward / forward by step |
| `[` / `]` | Jump backward / forward by 5× step |
| `,` / `.` | Decrease / increase step size |
| `G` | Go to a specific global frame (dialog) |
| `Space` | Play / pause |
| `R` | Run SAM2 on current frame only |
| `M` | Merge & propagate from current frame forward |
| `T` | Save masks (prompts for folder name) |
| `D` | Delete last point on current frame |
| `N` / `P` | Next / previous batch |

### Menu
| Menu item | Shortcut |
|-----------|----------|
| File → Open Video… | `Ctrl+O` |
| File → Open Folder of Frames… | `Ctrl+Shift+O` |
| File → Save Masks… | `Ctrl+S` |

## Object Categories

| # | Object | Colour |
|---|--------|--------|
| 1 | Head Surgeon | Green |
| 2 | Assistant Surgeon | Blue |
| 3 | Nurse | Red |
| 4 | Tool 1 | Orange |
| 5 | Tool 2 | Purple |

To change the objects, edit `OBJ_NAMES` and `OBJ_COLORS_BGR` at the top of `annotation_tool.py`.

## Output

### Saving Masks
Press `T` or use **File → Save Masks…** to choose an output folder name (default: `output_video_masks/` next to the script).  
Masks are also auto-saved to `output_video_masks/` whenever you advance to the next batch with `N`.

### File Format
- **Filename**: `frame_XXXXX_combined.png`
- **Format**: PNG, colour-coded by object
- **Frame numbers**: global across all batches (e.g. batch 2 starts at `frame_00100_…`)

## Troubleshooting

### SAM2 checkpoint not found
Set the `SAM2_ROOT` environment variable:
```bash
export SAM2_ROOT=/path/to/segment-anything-2
python annotation_tool.py
```

### "CUDA out of memory"
The tool is tested at Full HD (1920×1080). For 4K video, reduce `BATCH_SIZE` at the top of the script.

### Masks look correct on screen but saved files are blank
All saved masks are read from the same data the viewer uses (`objects_data`), so what you see is what is saved.

### Slow propagation
SAM2 saves the current batch frames to `./frames_from_video/` before propagation (~3–5 s for 100 frames, then ~80 s propagation at ~1.3 it/s on a Quadro RTX 6000).

## Technical Notes

- **Image predictor** (`R` key): `SAM2ImagePredictor` — fast, single frame
- **Video predictor** (`M` key): `build_sam2_video_predictor` — full batch propagation
- **Batch boundary**: last frame's masks from batch N are used as initialisation prompts (via `add_new_mask`) for batch N+1
- **Mask storage**: boolean arrays (H×W) in `objects_data["masks"]`; separate `video_segments` dict used during propagation

## Citation

If you use this tool in your research, please cite SAM2:

```bibtex
@article{ravi2024sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and Hu, Ronghang
          and Ryali, Chaitanya and Ma, Tengyu and Khedr, Haitham and R{\"a}dle, Roman
          and Rolland, Chloe and Gustafson, Laura and Mintun, Eric and Pan, Junting
          and Alwala, Kalyan Vasudev and Carion, Nicolas and Wu, Chao-Yuan
          and Girshick, Ross and Doll{\'a}r, Piotr and Feichtenhofer, Christoph},
  journal={arXiv preprint arXiv:2408.00714},
  year={2024}
}
```

---

**Last Updated**: June 2026  
**Tested On**: Ubuntu 22.04, CUDA 11.4, Quadro RTX 6000 (24 GB)
