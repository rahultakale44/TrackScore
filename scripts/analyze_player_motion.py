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
    PlayerMotionAnalysisError,
    PlayerMotionAnalyzer,
)

from backend.app.vision import (
    CourtHomography,
    CourtHomographyError,
    PlayerTracker,
    PlayerTrackerConfig,
    PlayerTrackingError,
    VideoLoader,
    VideoLoaderError,
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Track tennis players and calculate "
            "real-world movement analytics."
        )
    )

    parser.add_argument(
        "video_path",
        type=str,
        help="Input tennis match video.",
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
        help=(
            "Four court corner pixel coordinates: "
            "far-left, far-right, near-left, near-right."
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
        default=5.0,
        help="Maximum duration to process.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=(
            "outputs/player_motion/"
            "player_motion.mp4"
        ),
    )

    parser.add_argument(
        "--metadata",
        type=str,
        default=(
            "outputs/player_motion/"
            "player_motion.json"
        ),
    )

    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
    )

    return parser.parse_args()


def build_image_corners(
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


def draw_player_analytics(
    frame,
    players,
):
    overlay = frame.copy()

    for player in players:
        bbox = player[
            "bbox"
        ]

        x1 = int(
            round(
                bbox["x1"]
            )
        )

        y1 = int(
            round(
                bbox["y1"]
            )
        )

        x2 = int(
            round(
                bbox["x2"]
            )
        )

        y2 = int(
            round(
                bbox["y2"]
            )
        )

        label = player[
            "player_label"
        ]

        position = player[
            "court_position"
        ]

        movement = player[
            "movement"
        ]

        cv2.rectangle(
            overlay,
            (
                x1,
                y1,
            ),
            (
                x2,
                y2,
            ),
            (
                0,
                255,
                0,
            ),
            2,
        )

        lines = [
            (
                f"{label} "
                f"| ID {player['track_id']}"
            ),
            (
                f"Pos: "
                f"{position['x_meters']:.2f}m, "
                f"{position['y_meters']:.2f}m"
            ),
            (
                f"Speed: "
                f"{movement['smoothed_speed_kmh']:.1f} km/h"
            ),
            (
                f"Distance: "
                f"{movement['total_distance_meters']:.1f} m"
            ),
            (
                f"Zone: "
                f"{position['zone']}"
            ),
        ]

        text_y = max(
            25,
            y1 - 90,
        )

        for index, text in enumerate(
            lines
        ):
            cv2.putText(
                overlay,
                text,
                (
                    x1,
                    text_y
                    + index * 22,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (
                    0,
                    255,
                    0,
                ),
                2,
                cv2.LINE_AA,
            )

    return overlay


def main():
    args = parse_arguments()

    try:
        loader = VideoLoader(
            args.video_path
        )

        video_metadata = (
            loader.get_metadata()
        )

        fps = float(
            video_metadata[
                "fps"
            ]
        )

        width = int(
            video_metadata[
                "width"
            ]
        )

        height = int(
            video_metadata[
                "height"
            ]
        )

        # --------------------------------------------------------
        # COURT CALIBRATION
        # --------------------------------------------------------

        image_corners = (
            build_image_corners(
                args.corners
            )
        )

        homography = (
            CourtHomography()
        )

        calibration = (
            homography.calibrate(
                image_points=(
                    image_corners
                ),
                court_type=(
                    args.court_type
                ),
            )
        )

        # --------------------------------------------------------
        # PLAYER TRACKER
        # --------------------------------------------------------

        tracker = PlayerTracker(
            PlayerTrackerConfig(
                model_path=args.model
            )
        )

        motion_analyzer = (
            PlayerMotionAnalyzer(
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

            raise PlayerMotionAnalysisError(
                "Unable to create output video."
            )

        frame_number = 0
        frame_results = []

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

            analysed_players = (
                motion_analyzer
                .analyse_players(
                    tracking_result[
                        "players"
                    ]
                )
            )

            overlay = (
                draw_player_analytics(
                    frame,
                    analysed_players,
                )
            )

            cv2.putText(
                overlay,
                (
                    "TrackScore | "
                    f"Court: "
                    f"{calibration.court_length_meters:.2f}m x "
                    f"{calibration.court_width_meters:.2f}m"
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
                    "players": (
                        analysed_players
                    ),
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
            "video": video_metadata,
            "court": {
                "type": (
                    args.court_type
                ),
                "length_meters": (
                    calibration
                    .court_length_meters
                ),
                "width_meters": (
                    calibration
                    .court_width_meters
                ),
                "reprojection_error": (
                    calibration
                    .reprojection_error
                ),
                "image_corners": (
                    image_corners
                ),
            },
            "player_summary": (
                motion_analyzer
                .get_summary()
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
            "\nTrackScore Player Motion Analytics"
        )

        print("=" * 65)

        print(
            f"Frames processed: "
            f"{frame_number}"
        )

        print(
            f"Court: "
            f"{calibration.court_length_meters:.2f}m x "
            f"{calibration.court_width_meters:.2f}m"
        )

        print(
            "\nPlayer Summary:"
        )

        print(
            json.dumps(
                motion_analyzer
                .get_summary(),
                indent=4,
            )
        )

        print(
            f"\nOutput Video: "
            f"{output_path.resolve()}"
        )

        print(
            f"Metadata: "
            f"{metadata_path.resolve()}"
        )

        print("=" * 65)

        print(
            "Player motion analytics successful."
        )

    except (
        VideoLoaderError,
        PlayerTrackingError,
        CourtHomographyError,
        PlayerMotionAnalysisError,
        ValueError,
    ) as error:

        print(
            f"\nPlayer motion analysis failed: "
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