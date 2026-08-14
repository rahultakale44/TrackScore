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
    BallDetectionError,
    BallDetector,
    BallDetectorConfig,
    VideoLoader,
    VideoLoaderError,
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Detect tennis-ball candidates "
            "from a match video frame."
        )
    )

    parser.add_argument(
        "video_path",
        type=str,
        help="Input tennis video.",
    )

    parser.add_argument(
        "--timestamp",
        type=float,
        default=3.0,
        help="Video timestamp in seconds.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/ball_detection",
    )

    return parser.parse_args()


def read_frame(
    video_path: str,
    timestamp: float,
):
    if timestamp < 0:
        raise ValueError(
            "Timestamp cannot be negative."
        )

    loader = VideoLoader(
        video_path
    )

    metadata = (
        loader.get_metadata()
    )

    duration = float(
        metadata[
            "duration_seconds"
        ]
    )

    if timestamp > duration:
        raise ValueError(
            f"Timestamp {timestamp}s exceeds "
            f"video duration {duration}s."
        )

    capture = (
        loader.open_video()
    )

    try:
        capture.set(
            cv2.CAP_PROP_POS_MSEC,
            timestamp * 1000,
        )

        success, frame = (
            capture.read()
        )

        if not success or frame is None:
            raise BallDetectionError(
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

    try:
        (
            frame,
            frame_number,
            metadata,
        ) = read_frame(
            args.video_path,
            args.timestamp,
        )

        detector = BallDetector(
            BallDetectorConfig(
                model_path=args.model,
                confidence_threshold=(
                    args.confidence
                ),
            )
        )

        result = (
            detector.detect_ball(
                frame
            )
        )

        overlay = (
            detector.draw_detection(
                frame,
                result,
            )
        )

        output_dir = Path(
            args.output_dir
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        image_path = (
            detector.save_image(
                overlay,
                str(
                    output_dir
                    / "ball_detection.jpg"
                ),
            )
        )

        json_path = (
            output_dir
            / "ball_detection.json"
        )

        payload = {
            "video": {
                "filename": (
                    metadata[
                        "filename"
                    ]
                ),
                "resolution": (
                    metadata[
                        "resolution"
                    ]
                ),
                "fps": (
                    metadata[
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
            "\nTrackScore Ball Detection"
        )

        print("=" * 60)

        print(
            f"Frame: "
            f"{frame_number}"
        )

        print(
            f"Timestamp: "
            f"{args.timestamp}s"
        )

        print(
            f"Raw candidates: "
            f"{result['raw_candidate_count']}"
        )

        print(
            f"Filtered candidates: "
            f"{result['filtered_candidate_count']}"
        )

        print(
            f"Ball detected: "
            f"{result['ball_detected']}"
        )

        if result[
            "ball"
        ] is not None:

            print(
                "\nBest candidate:"
            )

            print(
                json.dumps(
                    result[
                        "ball"
                    ],
                    indent=4,
                )
            )

        print(
            f"\nOverlay: "
            f"{image_path.resolve()}"
        )

        print(
            f"JSON: "
            f"{json_path.resolve()}"
        )

        print("=" * 60)

    except (
        BallDetectionError,
        VideoLoaderError,
        ValueError,
    ) as error:

        print(
            f"\nBall detection failed: "
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