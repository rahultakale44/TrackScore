from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from backend.app.vision import (
    CourtLineDetectionError,
    CourtLineDetector,
    VideoLoader,
    VideoLoaderError,
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Detect candidate tennis court lines "
            "from a TrackScore match video."
        )
    )

    parser.add_argument(
        "video_path",
        type=str,
        help="Path to tennis match video.",
    )

    parser.add_argument(
        "--timestamp",
        type=float,
        default=10.0,
        help=(
            "Video timestamp in seconds. "
            "Default: 10"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/court_lines",
        help=(
            "Output directory for masks, edges, "
            "overlay, and JSON."
        ),
    )

    return parser.parse_args()


def read_frame(
    video_path: str,
    timestamp_seconds: float,
):
    if timestamp_seconds < 0:
        raise ValueError(
            "Timestamp cannot be negative."
        )

    loader = VideoLoader(video_path)

    metadata = loader.get_metadata()

    duration = float(
        metadata["duration_seconds"]
    )

    if timestamp_seconds > duration:
        raise ValueError(
            f"Timestamp {timestamp_seconds}s exceeds "
            f"video duration {duration}s."
        )

    capture = loader.open_video()

    try:
        capture.set(
            cv2.CAP_PROP_POS_MSEC,
            timestamp_seconds * 1000.0,
        )

        success, frame = capture.read()

        if not success or frame is None:
            raise CourtLineDetectionError(
                "Unable to read requested video frame."
            )

        frame_number = int(
            capture.get(
                cv2.CAP_PROP_POS_FRAMES
            )
        ) - 1

        return (
            frame,
            frame_number,
            metadata,
        )

    finally:
        capture.release()


def main():
    args = parse_arguments()

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        (
            frame,
            frame_number,
            video_metadata,
        ) = read_frame(
            args.video_path,
            args.timestamp,
        )

        detector = CourtLineDetector()

        line_mask = (
            detector.enhance_white_lines(
                frame
            )
        )

        edge_map = (
            detector.create_edge_map(
                frame
            )
        )

        lines = detector.detect_lines(
            frame
        )

        summary = detector.summarize_lines(
            lines
        )

        overlay = (
            detector.draw_debug_overlay(
                frame,
                lines,
            )
        )

        mask_path = detector.save_image(
            line_mask,
            str(
                output_dir
                / "court_line_mask.jpg"
            ),
        )

        edge_path = detector.save_image(
            edge_map,
            str(
                output_dir
                / "court_edges.jpg"
            ),
        )

        overlay_path = detector.save_image(
            overlay,
            str(
                output_dir
                / "court_line_overlay.jpg"
            ),
        )

        json_path = (
            output_dir
            / "court_lines.json"
        )

        payload = {
            "video": {
                "filename": (
                    video_metadata[
                        "filename"
                    ]
                ),
                "resolution": (
                    video_metadata[
                        "resolution"
                    ]
                ),
                "fps": (
                    video_metadata[
                        "fps"
                    ]
                ),
            },
            "selected_frame": {
                "frame_number": (
                    frame_number
                ),
                "timestamp_seconds": (
                    args.timestamp
                ),
            },
            "summary": summary,
            "lines": lines,
        }

        with json_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                payload,
                file,
                indent=4,
            )

        print(
            "\nTrackScore Court Line Detection"
        )

        print("=" * 55)

        print(
            f"Video: "
            f"{video_metadata['filename']}"
        )

        print(
            f"Timestamp: "
            f"{args.timestamp}s"
        )

        print(
            f"Frame number: "
            f"{frame_number}"
        )

        print(
            f"Total candidate lines: "
            f"{summary['total_lines']}"
        )

        print(
            f"Horizontal-like: "
            f"{summary['horizontal_lines']}"
        )

        print(
            f"Vertical-like: "
            f"{summary['vertical_lines']}"
        )

        print(
            f"Diagonal: "
            f"{summary['diagonal_lines']}"
        )

        print(
            f"Line mask: "
            f"{mask_path.resolve()}"
        )

        print(
            f"Edge map: "
            f"{edge_path.resolve()}"
        )

        print(
            f"Overlay: "
            f"{overlay_path.resolve()}"
        )

        print(
            f"JSON: "
            f"{json_path.resolve()}"
        )

        print("=" * 55)

        print(
            "Court-line candidate detection successful."
        )

    except (
        CourtLineDetectionError,
        VideoLoaderError,
        ValueError,
    ) as error:
        print(
            f"\nCourt-line detection failed: "
            f"{error}"
        )

        sys.exit(1)

    except Exception as error:
        print(
            f"\nUnexpected error: "
            f"{error}"
        )

        sys.exit(1)


if __name__ == "__main__":
    main()