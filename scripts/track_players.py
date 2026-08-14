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
    PlayerTracker,
    PlayerTrackerConfig,
    PlayerTrackingError,
    VideoLoader,
    VideoLoaderError,
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Track tennis players across a match video."
        )
    )

    parser.add_argument(
        "video_path",
        type=str,
        help="Input tennis video.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=(
            "outputs/player_tracking/"
            "tracked_players.mp4"
        ),
        help="Annotated output video.",
    )

    parser.add_argument(
        "--metadata",
        type=str,
        default=(
            "outputs/player_tracking/"
            "player_tracks.json"
        ),
        help="Tracking metadata JSON.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.30,
    )

    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help=(
            "Optional processing limit "
            "for development testing."
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

            raise PlayerTrackingError(
                "Unable to create output video."
            )

        tracker = PlayerTracker(
            PlayerTrackerConfig(
                model_path=args.model,
                confidence_threshold=(
                    args.confidence
                ),
            )
        )

        frame_number = 0

        processed_frames = 0

        while True:
            success, frame = (
                capture.read()
            )

            if not success:
                break

            timestamp_seconds = (
                frame_number / fps
            )

            if (
                args.max_seconds
                is not None
                and timestamp_seconds
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
                        timestamp_seconds
                    ),
                )
            )

            overlay = (
                tracker.draw_tracks(
                    frame,
                    result["players"],
                )
            )

            overlay = (
                tracker.draw_history(
                    overlay
                )
            )

            cv2.putText(
                overlay,
                (
                    f"Frame: "
                    f"{frame_number}"
                ),
                (25, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                overlay,
                (
                    f"Time: "
                    f"{timestamp_seconds:.2f}s"
                ),
                (25, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            writer.write(
                overlay
            )

            frame_number += 1
            processed_frames += 1

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
                processed_frames
            ),
            "track_to_player_label": (
                tracker
                .track_to_player_label
            ),
            "player_history": (
                tracker
                .player_history
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
            "\nTrackScore Player Tracking"
        )

        print("=" * 60)

        print(
            f"Processed frames: "
            f"{processed_frames}"
        )

        print(
            f"Output video: "
            f"{output_path.resolve()}"
        )

        print(
            f"Tracking metadata: "
            f"{metadata_path.resolve()}"
        )

        print(
            "\nPlayer Track Mapping:"
        )

        print(
            json.dumps(
                tracker
                .track_to_player_label,
                indent=4,
            )
        )

        print("=" * 60)

        print(
            "Player tracking successful."
        )

    except (
        PlayerTrackingError,
        VideoLoaderError,
        ValueError,
    ) as error:
        print(
            f"\nPlayer tracking failed: "
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