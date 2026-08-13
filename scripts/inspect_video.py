import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from backend.app.vision import VideoLoader, VideoLoaderError


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Inspect a tennis match video and display "
            "its metadata."
        )
    )

    parser.add_argument(
        "video_path",
        type=str,
        help="Path to the tennis match video.",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    try:
        loader = VideoLoader(args.video_path)

        metadata = loader.get_metadata()

        print("\nTrackScore Video Inspection")
        print("=" * 40)
        print(json.dumps(metadata, indent=4))
        print("=" * 40)
        print("Video validation successful.")

    except VideoLoaderError as error:
        print(f"\nVideo validation failed: {error}")
        sys.exit(1)

    except Exception as error:
        print(f"\nUnexpected error: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()