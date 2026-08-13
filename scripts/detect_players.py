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
    PlayerDetectionError,
    PlayerDetector,
    PlayerDetectorConfig,
    VideoLoader,
    VideoLoaderError,
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Detect tennis players from a TrackScore match video."
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
        help="Frame timestamp in seconds.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help="Ultralytics YOLO model path.",
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.30,
        help="Detection confidence threshold.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/player_detection",
        help="Output directory.",
    )

    return parser.parse_args()


def read_frame_at_timestamp(
    video_path: str,
    timestamp_seconds: float,
):
    if timestamp_seconds < 0:
        raise ValueError(
            "Timestamp cannot be negative."
        )

    loader = VideoLoader(
        video_path
    )

    metadata = loader.get_metadata()

    duration = float(
        metadata[
            "duration_seconds"
        ]
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
            raise PlayerDetectionError(
                "Unable to read requested frame."
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
        ) = read_frame_at_timestamp(
            args.video_path,
            args.timestamp,
        )

        config = PlayerDetectorConfig(
            model_path=args.model,
            confidence_threshold=(
                args.confidence
            ),
        )

        detector = PlayerDetector(
            config
        )

        result = (
            detector.detect_players(
                frame
            )
        )

        overlay = (
            detector.draw_players(
                frame,
                result["players"],
            )
        )

        overlay_path = (
            detector.save_image(
                overlay,
                str(
                    output_dir
                    / "player_detection.jpg"
                ),
            )
        )

        json_path = (
            output_dir
            / "player_detection.json"
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
            "frame": {
                "frame_number": (
                    frame_number
                ),
                "timestamp_seconds": (
                    args.timestamp
                ),
            },
            "detector": {
                "model": (
                    args.model
                ),
                "confidence_threshold": (
                    args.confidence
                ),
            },
            "result": result,
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
            "\nTrackScore Player Detection"
        )

        print("=" * 60)

        print(
            f"Video: "
            f"{video_metadata['filename']}"
        )

        print(
            f"Frame: "
            f"{frame_number}"
        )

        print(
            f"Timestamp: "
            f"{args.timestamp}s"
        )

        print(
            f"Raw persons detected: "
            f"{result['raw_person_count']}"
        )

        print(
            f"Court candidates: "
            f"{result['court_candidate_count']}"
        )

        print(
            f"Players selected: "
            f"{result['selected_player_count']}"
        )

        for player in result[
            "players"
        ]:
            print(
                f"\n{player['player_label']}"
            )

            print(
                f"  Court side: "
                f"{player['court_side']}"
            )

            print(
                f"  Confidence: "
                f"{player['confidence']}"
            )

            print(
                f"  Foot point: "
                f"{player['foot_point']}"
            )

        print(
            f"\nOverlay: "
            f"{overlay_path.resolve()}"
        )

        print(
            f"JSON: "
            f"{json_path.resolve()}"
        )

        print("=" * 60)

        print(
            "Player detection successful."
        )

    except (
        PlayerDetectionError,
        VideoLoaderError,
        ValueError,
    ) as error:
        print(
            f"\nPlayer detection failed: "
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