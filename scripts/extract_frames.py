import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from backend.app.vision import (
    FrameExtractionError,
    FrameExtractor,
    VideoLoaderError,
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Extract frames from a tennis match video "
            "for TrackScore processing."
        )
    )

    parser.add_argument(
        "video_path",
        type=str,
        help="Path to the source tennis video.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="outputs/frames",
        help="Directory where extracted frames will be saved.",
    )

    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help=(
            "Extract every Nth frame. "
            "Default: 30"
        ),
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional maximum number of frames to extract.",
    )

    parser.add_argument(
        "--timestamps",
        type=float,
        nargs="*",
        default=None,
        help=(
            "Extract exact timestamps in seconds. "
            "Example: --timestamps 1 5.5 10"
        ),
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    try:
        extractor = FrameExtractor(
            video_path=args.video_path,
            output_dir=args.output,
        )

        if args.timestamps:
            extracted_frames = extractor.extract_at_timestamps(
                args.timestamps
            )
            mode = "timestamp extraction"
        else:
            extracted_frames = extractor.extract_every_n_frames(
                interval=args.interval,
                max_frames=args.max_frames,
            )
            mode = "interval extraction"

        metadata_path = extractor.save_metadata_json(
            extracted_frames
        )

        print("\nTrackScore Frame Extraction")
        print("=" * 50)
        print(f"Mode: {mode}")
        print(
            f"Frames extracted: "
            f"{len(extracted_frames)}"
        )
        print(
            f"Output directory: "
            f"{Path(args.output).resolve()}"
        )
        print(
            f"Metadata file: "
            f"{metadata_path.resolve()}"
        )

        if extracted_frames:
            print("\nFirst extracted frame:")
            print(
                json.dumps(
                    extracted_frames[0],
                    indent=4,
                )
            )

        print("=" * 50)
        print("Frame extraction successful.")

    except (
        FrameExtractionError,
        VideoLoaderError,
    ) as error:
        print(f"\nFrame extraction failed: {error}")
        sys.exit(1)

    except Exception as error:
        print(f"\nUnexpected error: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()