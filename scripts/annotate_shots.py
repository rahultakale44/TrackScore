#!/usr/bin/env python3
"""
Interactive shot annotation tool for TrackScore ML dataset.

Provides a simple UI to review clips and assign labels.
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2


class ShotAnnotator:
    """Interactive shot annotation tool."""
    
    VALID_LABELS = [
        "forehand",
        "backhand",
        "serve",
        "volley",
        "smash",
        "drop",
        "unknown"
    ]
    
    def __init__(self, annotation_csv: str, clips_dir: str):
        self.annotation_csv = Path(annotation_csv)
        self.clips_dir = Path(clips_dir)
        
        # Load existing annotations
        self.annotations = []
        if self.annotation_csv.exists():
            with open(self.annotation_csv, 'r') as f:
                reader = csv.DictReader(f)
                self.annotations = list(reader)
        
        print(f"✓ Loaded {len(self.annotations)} shot candidates")
        
        # Count labeled vs unlabeled
        labeled = sum(1 for a in self.annotations if a["label"] != "unlabeled")
        print(f"  Labeled: {labeled}")
        print(f"  Unlabeled: {len(self.annotations) - labeled}")
    
    def annotate_interactive(self, annotator_name: str, start_index: int = 0):
        """Interactive annotation session."""
        print("\n" + "="*70)
        print("INTERACTIVE ANNOTATION SESSION")
        print("="*70)
        print("\nControls:")
        print("  1-7: Assign label (1=forehand, 2=backhand, 3=serve, 4=volley, 5=smash, 6=drop, 7=unknown)")
        print("  SPACE: Skip to next")
        print("  r: Replay clip")
        print("  b: Go back one clip")
        print("  s: Save and quit")
        print("  q: Quit without saving")
        print("="*70)
        
        current_idx = start_index
        modified = False
        
        while current_idx < len(self.annotations):
            annotation = self.annotations[current_idx]
            clip_path = self.clips_dir / f"{annotation['shot_id']}.mp4"
            
            if not clip_path.exists():
                print(f"\n⚠ Clip not found: {clip_path}")
                current_idx += 1
                continue
            
            # Display current annotation info
            print(f"\n[{current_idx + 1}/{len(self.annotations)}] {annotation['shot_id']}")
            print(f"  Video: {annotation['video']}")
            print(f"  Frames: {annotation['start_frame']}-{annotation['end_frame']}")
            print(f"  Current label: {annotation['label']}")
            print(f"  Suggested player: {annotation.get('suggested_player', 'unknown')}")
            
            # Play clip
            action = self._play_clip_and_get_input(clip_path)
            
            if action == 'quit':
                print("\n⚠ Quitting without saving")
                break
            elif action == 'save':
                self._save_annotations()
                print("\n✓ Annotations saved. Exiting.")
                break
            elif action == 'back':
                if current_idx > 0:
                    current_idx -= 1
                continue
            elif action == 'skip':
                current_idx += 1
                continue
            elif action == 'replay':
                continue
            elif action in self.VALID_LABELS:
                # Assign label
                annotation['label'] = action
                annotation['labeled_by'] = annotator_name
                modified = True
                print(f"✓ Labeled as: {action}")
                current_idx += 1
            else:
                print(f"⚠ Invalid action: {action}")
        
        if modified:
            save_choice = input("\nSave changes? (y/n): ").strip().lower()
            if save_choice == 'y':
                self._save_annotations()
                print("✓ Annotations saved")
    
    def _play_clip_and_get_input(self, clip_path: Path) -> str:
        """Play clip in OpenCV window and get user input."""
        cap = cv2.VideoCapture(str(clip_path))
        
        window_name = "Shot Annotation - Press key when ready"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # Play clip in loop until user makes choice
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            frames.append(frame)
            break
        
        # Read all frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        cap.release()
        
        if not frames:
            cv2.destroyWindow(window_name)
            return 'skip'
        
        # Display instructions on first frame
        instruction_frame = frames[0].copy()
        y_pos = 80
        cv2.rectangle(instruction_frame, (10, 60), (800, 350), (0, 0, 0), -1)
        cv2.putText(instruction_frame, "CONTROLS:", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 35
        cv2.putText(instruction_frame, "1=forehand  2=backhand  3=serve", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_pos += 30
        cv2.putText(instruction_frame, "4=volley  5=smash  6=drop  7=unknown", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_pos += 35
        cv2.putText(instruction_frame, "SPACE=skip  r=replay  b=back", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_pos += 30
        cv2.putText(instruction_frame, "s=save&quit  q=quit", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imshow(window_name, instruction_frame)
        cv2.waitKey(1500)  # Show instructions for 1.5 seconds
        
        # Play clip in loop
        action = None
        frame_idx = 0
        
        while action is None:
            cv2.imshow(window_name, frames[frame_idx])
            
            key = cv2.waitKey(30) & 0xFF
            
            if key == ord('1'):
                action = 'forehand'
            elif key == ord('2'):
                action = 'backhand'
            elif key == ord('3'):
                action = 'serve'
            elif key == ord('4'):
                action = 'volley'
            elif key == ord('5'):
                action = 'smash'
            elif key == ord('6'):
                action = 'drop'
            elif key == ord('7'):
                action = 'unknown'
            elif key == ord(' '):
                action = 'skip'
            elif key == ord('r'):
                action = 'replay'
            elif key == ord('b'):
                action = 'back'
            elif key == ord('s'):
                action = 'save'
            elif key == ord('q'):
                action = 'quit'
            elif key == 27:  # ESC
                action = 'quit'
            
            frame_idx = (frame_idx + 1) % len(frames)
        
        cv2.destroyWindow(window_name)
        return action
    
    def _save_annotations(self):
        """Save annotations back to CSV."""
        fieldnames = list(self.annotations[0].keys()) if self.annotations else []
        
        with open(self.annotation_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.annotations)
    
    def print_summary(self):
        """Print annotation summary statistics."""
        print("\n" + "="*70)
        print("ANNOTATION SUMMARY")
        print("="*70)
        
        label_counts = {}
        for annotation in self.annotations:
            label = annotation['label']
            label_counts[label] = label_counts.get(label, 0) + 1
        
        total = len(self.annotations)
        print(f"\nTotal samples: {total}")
        print("\nLabel distribution:")
        for label in sorted(label_counts.keys()):
            count = label_counts[label]
            pct = 100 * count / total
            print(f"  {label:12s}: {count:3d} ({pct:5.1f}%)")
        
        labeled = total - label_counts.get('unlabeled', 0)
        print(f"\nLabeling progress: {labeled}/{total} ({100*labeled/total:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Interactive shot annotation tool for TrackScore ML"
    )
    parser.add_argument(
        "--annotation-csv",
        default="data/shot_dataset/annotations/shot_annotations.csv",
        help="Path to annotation CSV file"
    )
    parser.add_argument(
        "--clips-dir",
        default="data/shot_dataset/clips",
        help="Directory containing video clips"
    )
    parser.add_argument(
        "--annotator",
        required=True,
        help="Name of annotator (will be recorded in labeled_by field)"
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start annotation from this index (to resume)"
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only print summary, don't start annotation session"
    )
    
    args = parser.parse_args()
    
    annotator = ShotAnnotator(args.annotation_csv, args.clips_dir)
    
    if args.summary_only:
        annotator.print_summary()
    else:
        annotator.annotate_interactive(args.annotator, args.start_index)
        annotator.print_summary()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
