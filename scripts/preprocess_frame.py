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
    FramePreprocessingError,
    FramePreprocessor,
    VideoLoader,
    VideoLoaderError,
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Extract and preprocess a tennis video frame "
            "for TrackScore machine-learning models."
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
        default=0.0,
        help=(
            "Timestamp in seconds of the frame "
            "to preprocess. Default: 0"
        ),
    )

    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="ML target width. Default: 640",
    )

    parser.add_argument(
        "--height",
        type=int,
        default=640,
        help="ML target height. Default: 640",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="outputs/preprocessed/frame.jpg",
        help="Processed preview output path.",
    )

    parser.add_argument(
        "--metadata",
        type=str,
        default="outputs/preprocessed/metadata.json",
        help="Metadata JSON output path.",
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
            timestamp_seconds * 1000,
        )

        success, frame = capture.read()

        if not success or frame is None:
            raise FramePreprocessingError(
                "Unable to read requested video frame."
            )

        frame_number = int(
            capture.get(
                cv2.CAP_PROP_POS_FRAMES
            )
        ) - 1

        return frame, frame_number, metadata

    finally:
        capture.release()


def main():
    args = parse_arguments()

    try:
        frame, frame_number, video_metadata = (
            read_frame_at_timestamp(
                video_path=args.video_path,
                timestamp_seconds=args.timestamp,
            )
        )

        preprocessor = FramePreprocessor(
            target_width=args.width,
            target_height=args.height,
        )

        processed_frame, preprocessing_metadata = (
            preprocessor.process_frame(frame)
        )

        output_path = (
            preprocessor.save_processed_frame(
                processed_frame,
                args.output,
            )
        )

        metadata_path = Path(args.metadata)

        metadata_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        complete_metadata = {
            "video": {
                "filename": video_metadata["filename"],
                "resolution": video_metadata["resolution"],
                "fps": video_metadata["fps"],
                "duration_seconds": (
                    video_metadata["duration_seconds"]
                ),
            },
            "selected_frame": {
                "frame_number": frame_number,
                "timestamp_seconds": args.timestamp,
            },
            "preprocessing": preprocessing_metadata,
        }

        with metadata_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                complete_metadata,
                file,
                indent=4,
            )

        print("\nTrackScore Frame Preprocessing")
        print("=" * 50)

        print(
            f"Video: "
            f"{video_metadata['filename']}"
        )

        print(
            f"Frame: {frame_number}"
        )

        print(
            f"Timestamp: {args.timestamp}s"
        )

        print(
            f"Input resolution: "
            f"{video_metadata['resolution']}"
        )

        print(
            f"ML resolution: "
            f"{args.width}x{args.height}"
        )

        quality = preprocessing_metadata[
            "quality"
        ]

        print(
            f"Brightness: "
            f"{quality['brightness']}"
        )

        print(
            f"Sharpness: "
            f"{quality['sharpness']}"
        )

        print(
            f"Frame acceptable: "
            f"{quality['acceptable']}"
        )

        print(
            f"Quality issues: "
            f"{quality['issues']}"
        )

        print(
            f"Model shape: "
            f"{preprocessing_metadata['model_input']['shape']}"
        )

        print(
            f"Preview saved: "
            f"{output_path.resolve()}"
        )

        print(
            f"Metadata saved: "
            f"{metadata_path.resolve()}"
        )

        print("=" * 50)

        print(
            "Frame preprocessing successful."
        )

    except (
        VideoLoaderError,
        FramePreprocessingError,
        ValueError,
    ) as error:
        print(
            f"\nFrame preprocessing failed: {error}"
        )
        sys.exit(1)

    except Exception as error:
        print(
            f"\nUnexpected error: {error}"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()