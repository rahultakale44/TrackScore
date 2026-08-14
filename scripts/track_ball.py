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
    BallTracker,
    BallTrackerConfig,
    BallTrackingError,
    VideoLoader,
    VideoLoaderError,
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Track a tennis ball temporally "
            "across a tennis match video."
        )
    )

    parser.add_argument(
        "video_path",
        type=str,
        help="Input tennis match video.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help="Ultralytics model.",
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.05,
        help="Ball detection confidence.",
    )

    parser.add_argument(
        "--max-seconds",
        type=float,
        default=5.0,
        help="Development processing duration.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=(
            "outputs/ball_tracking/"
            "tracked_ball.mp4"
        ),
    )

    parser.add_argument(
        "--metadata",
        type=str,
        default=(
            "outputs/ball_tracking/"
            "ball_track.json"
        ),
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    try:
        loader = VideoLoader(
            args.video_path
        )

        metadata = (
            loader.get_metadata()
        )

        fps = float(
            metadata["fps"]
        )

        width = int(
            metadata["width"]
        )

        height = int(
            metadata["height"]
        )

        tracker = BallTracker(
            BallTrackerConfig(
                model_path=args.model,
                confidence_threshold=(
                    args.confidence
                ),
            )
        )

        capture = (
            loader.open_video()
        )

        output_path = Path(
            args.output
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(
                *"mp4v"
            ),
            fps,
            (
                width,
                height,
            ),
        )

        if not writer.isOpened():
            capture.release()

            raise BallTrackingError(
                "Unable to create output tracking video."
            )

        frame_number = 0

        frame_results = []

        visible_count = 0
        prediction_count = 0

        while True:
            success, frame = (
                capture.read()
            )

            if (
                not success
                or frame is None
            ):
                break

            timestamp = (
                frame_number
                / fps
            )

            if (
                args.max_seconds
                is not None
                and timestamp
                > args.max_seconds
            ):
                break

            result = (
                tracker.process_frame(
                    frame=frame,
                    frame_number=(
                        frame_number
                    ),
                    timestamp_seconds=(
                        timestamp
                    ),
                )
            )

            if result[
                "ball_visible"
            ]:
                visible_count += 1

            ball = result.get(
                "ball"
            )

            if (
                ball is not None
                and ball.get(
                    "predicted",
                    False,
                )
            ):
                prediction_count += 1

            overlay = (
                tracker.draw_tracking(
                    frame,
                    result,
                )
            )

            cv2.putText(
                overlay,
                (
                    f"TrackScore | "
                    f"Frame {frame_number}"
                ),
                (
                    25,
                    40,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (
                    255,
                    255,
                    255,
                ),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                overlay,
                (
                    f"Time: "
                    f"{timestamp:.2f}s"
                ),
                (
                    25,
                    75,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (
                    255,
                    255,
                    255,
                ),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                overlay,
                (
                    f"YOLO candidates: "
                    f"{result['yolo_candidate_count']}"
                ),
                (
                    25,
                    110,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (
                    255,
                    255,
                    255,
                ),
                2,
                cv2.LINE_AA,
            )

            writer.write(
                overlay
            )

            frame_results.append(
                {
                    "frame_number": (
                        frame_number
                    ),

                    "timestamp_seconds": round(
                        timestamp,
                        3,
                    ),

                    "ball_visible": (
                        result[
                            "ball_visible"
                        ]
                    ),

                    "track_active": (
                        result[
                            "track_active"
                        ]
                    ),

                    "missed_frames": (
                        result[
                            "missed_frames"
                        ]
                    ),

                    "ball": result[
                        "ball"
                    ],
                }
            )

            frame_number += 1

        capture.release()
        writer.release()

        metadata_path = Path(
            args.metadata
        )

        metadata_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {
            "video": metadata,

            "processed_frames": (
                frame_number
            ),

            "visible_detection_frames": (
                visible_count
            ),

            "prediction_frames": (
                prediction_count
            ),

            "tracking_summary": (
                tracker.get_summary()
            ),

            "trajectory": (
                tracker.history
            ),

            "frames": (
                frame_results
            ),
        }

        with metadata_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                payload,
                file,
                indent=4,
            )

        print(
            "\nTrackScore Temporal Ball Tracking"
        )

        print("=" * 65)

        print(
            f"Processed frames: "
            f"{frame_number}"
        )

        print(
            f"Direct detections: "
            f"{visible_count}"
        )

        print(
            f"Predicted frames: "
            f"{prediction_count}"
        )

        print(
            "\nTracking Summary:"
        )

        print(
            json.dumps(
                tracker.get_summary(),
                indent=4,
            )
        )

        print(
            f"\nOutput video: "
            f"{output_path.resolve()}"
        )

        print(
            f"Metadata: "
            f"{metadata_path.resolve()}"
        )

        print("=" * 65)

        print(
            "Temporal ball tracking successful."
        )

    except (
        VideoLoaderError,
        BallTrackingError,
        ValueError,
    ) as error:

        print(
            f"\nBall tracking failed: "
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