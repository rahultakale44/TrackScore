# TrackScore ML Shot Dataset

## Overview
This directory contains the tennis shot dataset for training ML shot classification models.

## Directory Structure
```
data/shot_dataset/
├── clips/                    # Video clips of candidate shots (1-2 second clips)
├── annotations/             # CSV files with manual labels
│   └── shot_annotations.csv # Main annotation file
├── raw_samples/            # Optional: full rally segments before clipping
└── README.md               # This file
```

## Target Shot Classes
- `forehand` - Forehand groundstroke
- `backhand` - Backhand groundstroke
- `serve` - Service motion
- `volley` - Volley (hit before bounce)
- `smash` - Overhead smash
- `drop` - Drop shot
- `unknown` - Cannot determine or other shot types

## Workflow

### Step 1: Extract Shot Candidates
Extract candidate shot moments from tennis video:

```bash
python scripts/extract_shot_samples.py samples/tennis_match3.mp4 \
    --calibration data/calibration/tennis_match3.json \
    --output-dir data/shot_dataset/clips \
    --annotation-csv data/shot_dataset/annotations/shot_annotations.csv
```

Options:
- `--max-clips N` - Limit to first N clips (for testing)

This will:
1. Analyze video to detect rally segments (both players visible)
2. Extract candidate shot clips (~1 second each)
3. Save clips as MP4 files in `clips/`
4. Generate annotation template CSV in `annotations/`

### Step 2: Manual Annotation

#### Option A: Interactive Tool (Recommended)
```bash
python scripts/annotate_shots.py \
    --annotation-csv data/shot_dataset/annotations/shot_annotations.csv \
    --clips-dir data/shot_dataset/clips \
    --annotator "YourName"
```

Controls:
- `1-7`: Assign label (1=forehand, 2=backhand, 3=serve, 4=volley, 5=smash, 6=drop, 7=unknown)
- `SPACE`: Skip to next
- `r`: Replay current clip
- `b`: Go back one clip
- `s`: Save and quit
- `q`: Quit without saving

To resume from specific clip:
```bash
python scripts/annotate_shots.py --annotator "YourName" --start-index 25
```

#### Option B: Spreadsheet Annotation
1. Open `data/shot_dataset/annotations/shot_annotations.csv` in Excel/LibreOffice
2. Watch each clip in `data/shot_dataset/clips/`
3. For each row:
   - Set `label` to one of the valid shot classes
   - Set `confidence` to: high, medium, or low
   - Set `labeled_by` to your name
   - Add any notes in `notes` column
4. Save the CSV file

### Step 3: Check Annotation Progress
```bash
python scripts/annotate_shots.py --summary-only
```

Output shows:
- Total samples
- Label distribution
- Labeling progress percentage

### Step 4: Feature Extraction & ML Training
(To be implemented in Phase 2-3)

## Annotation Schema

### CSV Fields
| Field | Type | Description | Required |
|-------|------|-------------|----------|
| `shot_id` | string | Unique identifier (e.g., shot_0001) | Yes |
| `video` | string | Source video filename | Yes |
| `rally_id` | int | Rally segment ID within video | Yes |
| `start_frame` | int | First frame of clip | Yes |
| `end_frame` | int | Last frame of clip | Yes |
| `center_frame` | int | Estimated shot moment frame | Yes |
| `duration_sec` | float | Clip duration in seconds | Yes |
| `suggested_player` | string | Heuristic guess (player_a/player_b/unknown) | No |
| `label` | string | **MANUAL**: Shot type (forehand/backhand/etc) | **Required for ML** |
| `confidence` | string | **MANUAL**: Annotation confidence (high/medium/low) | Recommended |
| `labeled_by` | string | **MANUAL**: Annotator name | Recommended |
| `notes` | string | **MANUAL**: Any additional observations | Optional |

**Important**: 
- `suggested_player` is a HEURISTIC guess from the extraction tool, NOT ground truth
- Only `label`, `confidence`, `labeled_by`, and `notes` should be manually filled
- Never modify `shot_id`, `start_frame`, `end_frame`, etc.

## Dataset Quality Guidelines

### Minimum Dataset Requirements
For reliable ML training, aim for:
- **Minimum 50 labeled samples** (absolute minimum)
- **Balanced classes**: At least 10 samples per class
- **Multiple annotators**: Inter-annotator agreement validation recommended

### Annotation Guidelines
1. **Watch full clip**: Don't decide from a single frame
2. **Focus on stroke type**: Classify the hitting motion, not the outcome
3. **Mark unclear as 'unknown'**: Don't guess if uncertain
4. **Note edge cases**: Use the notes field for ambiguous cases
5. **Consistency**: Try to apply the same criteria across all clips

### Class-Specific Notes
- **Forehand vs Backhand**: Based on which side of body player swings
- **Volley**: Ball is hit before bouncing (usually near net)
- **Smash**: Overhead shot with downward trajectory
- **Serve**: Overhand motion from baseline, ball tossed up first
- **Drop**: Soft shot aimed just over net (often with slice)
- **Unknown**: Can't determine, obscured view, or non-standard shot

## Current Status
- Extraction tool: ✓ Working
- Interactive annotator: ✓ Working
- Extracted clips: 8 samples from tennis_match3.mp4 (test)
- Labeled samples: 0
- Feature extraction: Not yet implemented
- ML training: Not yet implemented

## Next Steps
1. Extract more samples from additional tennis videos
2. Complete manual annotation (aim for 100+ labeled samples)
3. Implement feature extraction pipeline
4. Train classical ML baseline (RandomForest, GradientBoosting)
5. Evaluate model performance
6. Integrate trained model into trackscore.py

## Notes
- Do NOT create synthetic/fake labels for testing
- Only report ML metrics on real manually-labeled data
- Heuristic-based shot detection should be clearly marked as such
- Keep dataset version controlled (track annotation date, annotator, etc.)
