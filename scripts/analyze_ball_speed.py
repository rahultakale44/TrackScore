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


from backend.app.analytics import (
    BallSpeedAnalysisError,
    BallSpeedAnalyzer,
)

from backend.app.vision import (
    BallTracker,
    BallTrackerConfig,
    BallTrackingError,
    CourtHomography,
    CourtHomographyError,
    VideoLoader,
    VideoLoaderError,
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Estimate real-world tennis-ball speed "
            "from tracked video trajectory."
        )
    )

    parser.add_argument(
        "video_path",
        type=str,
    )

    parser.add_argument(
        "--corners",
        type=float,
        nargs=8,
        required=True,
        metavar=(
            "FL_X",
            "FL_Y",
            "FR_X",
            "FR_Y",
            "NL_X",
            "NL_Y",
            "NR_X",
            "NR_Y",
        ),
    )

    parser.add_argument(
        "--court-type",
        type=str,
        default="singles",
        choices=[
            "singles",
            "doubles",
        ],
    )

    parser.add_argument(
        "--max-seconds",
        type=float,
        default=8.0,
    )

    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=(
            "outputs/ball_speed/"
            "ball_speed.mp4"
        ),
    )

    parser.add_argument(
        "--metadata",
        type=str,
        default=(
            "outputs/ball_speed/"
            "ball_speed.json"
        ),
    )

    return parser.parse_args()


def build_corners(
    values,
):
    return [
        [
            values[0],
            values[1],
        ],
        [
            values[2],
            values[3],
        ],
        [
            values[4],
            values[5],
        ],
        [
            values[6],
            values[7],
        ],
    ]


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
            metadata[
                "fps"
            ]
        )

        width = int(
            metadata[
                "width"
            ]
        )

        height = int(
            metadata[
                "height"
            ]
        )

        corners = (
            build_corners(
                args.corners
            )
        )

        homography = (
            CourtHomography()
        )

        homography.calibrate(
            image_points=(
                corners
            ),
            court_type=(
                args.court_type
            ),
        )

        tracker = BallTracker(
            BallTrackerConfig(
                model_path=args.model
            )
        )

        speed_analyzer = (
            BallSpeedAnalyzer(
                homography
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

            raise BallSpeedAnalysisError(
                "Unable to create output video."
            )

        frame_number = 0

        analysed_points = []

        processed_history_count = 0

        latest_speed = 0.0
        peak_speed = 0.0

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

            tracking_result = (
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

            while (
                processed_history_count
                < len(
                    tracker.history
                )
            ):
                point = tracker.history[
                    processed_history_count
                ]

                speed_result = (
                    speed_analyzer
                    .analyse_point(
                        point
                    )
                )

                analysed_points.append(
                    speed_result
                )

                latest_speed = (
                    speed_result[
                        "smoothed_speed_kmh"
                    ]
                )

                peak_speed = (
                    speed_result[
                        "peak_speed_kmh"
                    ]
                )

                processed_history_count += 1

            overlay = (
                tracker.draw_tracking(
                    frame,
                    tracking_result,
                )
            )

            cv2.rectangle(
                overlay,
                (
                    20,
                    20,
                ),
                (
                    470,
                    145,
                ),
                (
                    0,
                    0,
                    0,
                ),
                -1,
            )

            cv2.putText(
                overlay,
                "TrackScore Ball Analytics",
                (
                    35,
                    50,
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
                    f"Estimated Speed: "
                    f"{latest_speed:.1f} km/h"
                ),
                (
                    35,
                    85,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (
                    0,
                    255,
                    255,
                ),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                overlay,
                (
                    f"Peak Speed: "
                    f"{peak_speed:.1f} km/h"
                ),
                (
                    35,
                    120,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (
                    0,
                    255,
                    0,
                ),
                2,
                cv2.LINE_AA,
            )

            writer.write(
                overlay
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

            "court": {
                "type": (
                    args.court_type
                ),
                "corners": corners,
            },

            "processed_frames": (
                frame_number
            ),

            "summary": (
                speed_analyzer
                .get_summary()
            ),

            "trajectory": (
                analysed_points
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
            "\nTrackScore Ball Speed Analysis"
        )

        print("=" * 65)

        print(
            f"Processed frames: "
            f"{frame_number}"
        )

        summary = (
            speed_analyzer
            .get_summary()
        )

        print(
            f"Trajectory points: "
            f"{summary['trajectory_points']}"
        )

        print(
            f"Average speed: "
            f"{summary['average_speed_kmh']:.2f} km/h"
        )

        print(
            f"Peak speed: "
            f"{summary['peak_speed_kmh']:.2f} km/h"
        )

        print(
            f"Total tracked distance: "
            f"{summary['total_distance_meters']:.2f} m"
        )

        print(
            "\nNote: speeds are court-plane estimates, "
            "not true 3D radar measurements."
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
            "Ball speed analysis successful."
        )

    except (
        VideoLoaderError,
        BallTrackingError,
        CourtHomographyError,
        BallSpeedAnalysisError,
        ValueError,
    ) as error:

        print(
            f"\nBall speed analysis failed: "
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