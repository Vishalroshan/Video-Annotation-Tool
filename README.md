# SAM2 Video Annotation Tool

A powerful interactive tool for annotating surgical videos using Meta's Segment Anything Model 2 (SAM2). This tool processes videos in 100-frame batches with automatic mask propagation across batch boundaries.

## Features

- **Batch Processing**: Processes videos in 100-frame chunks to manage GPU memory efficiently
- **Interactive Annotation**: Click-based point annotation for object segmentation
- **Smart Propagation**: Automatically propagates masks from previous batches
- **Frame-by-Frame Control**: Generate masks on individual frames or propagate through entire batches
- **Auto-Save**: Automatically saves masks when moving between batches
- **24 Pre-defined Objects**: Optimized for surgical operating room scenarios

## Requirements

### Hardware
- **GPU**: NVIDIA GPU with 24GB VRAM (tested on Quadro RTX 6000)
- **RAM**: 125GB system RAM recommended
- **Storage**: Sufficient space for video and output masks

### Software
- Python 3.10+
- CUDA 11.4+
- SAM2 installed from [facebookresearch/segment-anything-2](https://github.com/facebookresearch/segment-anything-2)

### Python Dependencies
```bash
pip install torch torchvision opencv-python pillow numpy
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

3. **Place the annotation script**:
```bash
# Copy vid_9.py to segment-anything-2/vis_sam2/
cp vid_9.py segment-anything-2/vis_sam2/
cd segment-anything-2/vis_sam2/
```

## Usage

### Basic Command
```bash
python vid_9.py <path_to_video.mp4>
```

Or with a folder of frames:
```bash
python vid_9.py <path_to_frames_folder>
```

### Workflow

#### Batch 1 (Initial Annotation)
1. **Add Points**: Left-click on objects in frame 0
2. **Generate Masks**: Press `R` (Single frame) or click "Single" button
3. **Propagate**: Press `M` or click "Batch" button to propagate through frames 0-99
4. **Move to Next Batch**: Press `N` (auto-saves masks)

#### Batch 2+ (With Previous Masks)
1. **Review Masks**: Previous batch's masks automatically appear on frame 0
2. **Option A - Good Masks**: Press `M` immediately to propagate
3. **Option B - Need Corrections**: 
   - Add correction points on any frame
   - Press `R` to generate mask for that frame
   - Navigate to other frames, repeat as needed
   - Press `M` to propagate from current frame forward
4. **Move to Next Batch**: Press `N` (auto-saves masks)

## Controls

### Mouse Controls
| Action | Description |
|--------|-------------|
| **Left Click** | Add positive point (include in mask) |
| **Middle Click** | Add negative point (exclude from mask) |
| **Shift + Click** | Remove nearest point |

### Keyboard Shortcuts
| Key | Action |
|-----|--------|
| **A / Z** | Switch to previous/next object |
| **; / '** | Navigate backward/forward 1 frame |
| **[ / ]** | Jump backward/forward 5 frames |
| **R** | Run SAM2 on current frame only |
| **M** | Merge & propagate from current frame forward |
| **T** | Manually save masks (optional, auto-saves on batch change) |
| **D** | Delete last point on current frame |
| **N / P** | Next/previous batch |
| **Q / ESC** | Quit application |

### On-Screen Buttons
- **<<5** / **<1** / **1>** / **5>>**: Frame navigation
- **Single**: Run SAM2 on current frame (same as `R`)
- **Batch**: Propagate from current frame forward (same as `M`)

## Object Categories

The tool supports 24 pre-defined surgical object categories:

| ID | Object | ID | Object |
|----|--------|----|--------|
| 1 | Patient | 13 | Drill |
| 2 | Anesthetist | 14 | Hammer |
| 3 | Assistant Surgeon | 15 | Instrument Table |
| 4 | Circulator | 16 | Tracker |
| 5 | Head Surgeon | 17 | Mako Robot |
| 6 | MPS | 18 | Monitor |
| 7 | Nurse | 19 | MPS Station |
| 8 | Student | 20 | OT |
| 9 | Unrelated Person | 21 | Saw |
| 10 | Instrument | 22 | Secondary Table |
| 11 | AE | 23 | Cementer |
| 12 | C-Arm | 24 | Drape |

## Output

### Default Location
```
segment-anything-2/vis_sam2/output_video_masks/
```

### File Format
- **Filename**: `frame_XXXXX_combined.png`
- **Format**: PNG images with color-coded masks
- **Naming**: Frame numbers are global across all batches

### Color Coding
Each object has a unique color for easy identification in the output masks.

## Advanced Features

### Frame-by-Frame Correction
If you notice errors in the middle of a batch:
1. Navigate to the problematic frame
2. Add correction points
3. Press `R` to generate mask for that frame only
4. Press `M` to propagate forward (preserves all frames before current)

### Forward-Only Propagation
When you press `M` (Merge & Propagate):
- Uses masks from current frame (or most recent annotated frame)
- Propagates **only forward** to end of batch
- **Preserves** all masks before current frame unchanged

### Memory Management
- **Batch Size**: 100 frames per batch
- **GPU Memory**: ~8-12 GB for 100 Full HD frames
- **Auto-cleanup**: Frames saved to disk are cleaned up between batches

## Troubleshooting

### "CUDA out of memory" Error
- **Cause**: Video resolution too high for GPU
- **Solution**: The tool is optimized for 100 frames at Full HD (1920×1080)

### "dtype mismatch" Error
- **Cause**: Mixed precision issue in SAM2
- **Solution**: Already fixed with `torch.autocast()` in the code

### Masks Not Propagating
- **Check**: Ensure you pressed `R` to generate initial masks
- **Check**: Press `M` to trigger propagation
- **Tip**: Watch for console messages confirming propagation

### Slow Performance
- **Frame Saving**: Initial frame saving takes 3-5 seconds (normal)
- **SAM2 Propagation**: ~1.3 it/s for 100 frames (~80 seconds total)
- **Single Frame**: ~1-2 seconds per frame

## Tips & Best Practices

1. **Start Conservative**: Begin with 1-2 positive points per object
2. **Use Negative Points**: Add negative points to exclude unwanted regions
3. **Check Frame 99**: Always review last frame before moving to next batch
4. **Incremental Corrections**: Use `R` for single-frame fixes rather than re-propagating entire batch
5. **Regular Saves**: Although auto-save is enabled, manually save important work with `T`
6. **Monitor GPU**: Watch `nvidia-smi` to ensure GPU memory is not maxing out

## Known Limitations

- **Video Length**: Very long videos (>20,000 frames) will have many batches
- **Object Appearance Changes**: Large appearance changes may require manual correction
- **Occlusions**: Heavy occlusions may need multiple correction points
- **Real-time**: Not designed for real-time annotation (processing time ~1-2 min per 100 frames)

## Technical Details

### Architecture
- **Image Predictor**: Used for single-frame mask generation (`R` key)
- **Video Predictor**: Used for batch propagation (`M` key)
- **Mask Format**: Boolean arrays (H×W) converted to uint8 for storage

### Batch Boundary Handling
- Last frame masks from batch N automatically copied to first frame of batch N+1
- Masks converted to initialization prompts for SAM2 video predictor
- Uses SAM2's `add_new_mask()` API for seamless propagation

### File Structure
```
segment-anything-2/
├── checkpoints/
│   └── sam2.1_hiera_large.pt
├── configs/
│   └── sam2.1/
│       └── sam2.1_hiera_l.yaml
└── vis_sam2/
    ├── vid_9.py (this script)
    ├── frames_from_video/ (temporary, auto-cleaned)
    └── output_video_masks/ (output)
```

## Citation

If you use this tool in your research, please cite SAM2:

```bibtex
@article{ravi2024sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and Hu, Ronghang and Ryali, Chaitanya and Ma, Tengyu and Khedr, Haitham and R{\"a}dle, Roman and Rolland, Chloe and Gustafson, Laura and Mintun, Eric and Pan, Junting and Alwala, Kalyan Vasudev and Carion, Nicolas and Wu, Chao-Yuan and Girshick, Ross and Doll{\'a}r, Piotr and Feichtenhofer, Christoph},
  journal={arXiv preprint arXiv:2408.00714},
  year={2024}
}
```

## Support

For issues related to:
- **This tool**: Check console output for error messages
- **SAM2**: Visit [SAM2 GitHub Issues](https://github.com/facebookresearch/segment-anything-2/issues)
- **CUDA/GPU**: Verify with `nvidia-smi` and check PyTorch CUDA compatibility

## License

This tool follows SAM2's Apache 2.0 License. See SAM2 repository for details.

---

**Version**: 1.0  
**Last Updated**: October 2025  
**Tested On**: Ubuntu 22.04, CUDA 11.4, Quadro RTX 6000 (24GB)
