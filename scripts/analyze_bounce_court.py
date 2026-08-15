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
    BallTrajectoryAnalyzer,
    BounceCourtAnalysisError,
    BounceCourtAnalyzer,
    BounceCourtConfig,
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
            "Track tennis ball, detect bounce candidates, "
            "map them to court coordinates, and classify IN/OUT."
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
            "outputs/bounce_court/"
            "bounce_court.mp4"
        ),
    )

    parser.add_argument(
        "--metadata",
        type=str,
        default=(
            "outputs/bounce_court/"
            "bounce_court.json"
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


def draw_bounce_analysis(
    frame,
    bounces,
):
    overlay = frame.copy()

    for bounce in bounces:
        position = bounce[
            "pixel_position"
        ]

        x = int(
            round(
                position["x"]
            )
        )

        y = int(
            round(
                position["y"]
            )
        )

        call = bounce[
            "line_call"
        ]

        if call == "IN":
            color = (
                0,
                255,
                0,
            )

        else:
            color = (
                0,
                0,
                255,
            )

        cv2.circle(
            overlay,
            (
                x,
                y,
            ),
            16,
            color,
            3,
        )

        court = bounce[
            "court_position"
        ]

        text = (
            f"{call} | "
            f"{court['x_meters']:.2f}m, "
            f"{court['y_meters']:.2f}m"
        )

        cv2.putText(
            overlay,
            text,
            (
                x + 20,
                max(
                    y - 12,
                    25,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
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

        image_corners = (
            build_corners(
                args.corners
            )
        )

        homography = (
            CourtHomography()
        )

        homography.calibrate(
            image_points=(
                image_corners
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

        trajectory_analyzer = (
            BallTrajectoryAnalyzer()
        )

        bounce_analyzer = (
            BounceCourtAnalyzer(
                homography=(
                    homography
                ),
                config=(
                    BounceCourtConfig(
                        court_type=(
                            args.court_type
                        )
                    )
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

            raise BounceCourtAnalysisError(
                "Unable to create output video."
            )

        frame_number = 0

        all_candidates = []

        analysed_bounces = []

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

            trajectory_result = (
                trajectory_analyzer
                .analyse_trajectory(
                    tracker.history
                )
            )

            current_candidates = (
                trajectory_result[
                    "bounce_candidates"
                ]
            )

            all_candidates.extend(
                current_candidates
            )

            unique_candidates = (
                trajectory_analyzer
                .deduplicate_bounces(
                    all_candidates
                )
            )

            analysed_bounces = (
                bounce_analyzer
                .analyse_bounces(
                    unique_candidates
                )
            )

            overlay = (
                tracker.draw_tracking(
                    frame,
                    tracking_result,
                )
            )

            overlay = (
                draw_bounce_analysis(
                    overlay,
                    analysed_bounces,
                )
            )

            summary = (
                bounce_analyzer
                .summarize(
                    analysed_bounces
                )
            )

            cv2.putText(
                overlay,
                (
                    f"IN: "
                    f"{summary['in_bounces']} "
                    f"| OUT: "
                    f"{summary['out_bounces']}"
                ),
                (
                    25,
                    110,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
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

            frame_number += 1

        capture.release()

        writer.release()

        summary = (
            bounce_analyzer
            .summarize(
                analysed_bounces
            )
        )

        metadata_path = Path(
            args.metadata
        )

        metadata_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {
            "video": (
                video_metadata
            ),

            "court": {
                "type": (
                    args.court_type
                ),
                "image_corners": (
                    image_corners
                ),
            },

            "processed_frames": (
                frame_number
            ),

            "summary": summary,

            "bounces": (
                analysed_bounces
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
            "\nTrackScore Bounce Court Analysis"
        )

        print("=" * 65)

        print(
            f"Processed frames: "
            f"{frame_number}"
        )

        print(
            f"Accepted bounces: "
            f"{summary['accepted_bounces']}"
        )

        print(
            f"IN: "
            f"{summary['in_bounces']}"
        )

        print(
            f"OUT: "
            f"{summary['out_bounces']}"
        )

        print(
            f"Near line: "
            f"{summary['near_line_bounces']}"
        )

        if analysed_bounces:
            print(
                "\nBounce Events:"
            )

            for bounce in (
                analysed_bounces
            ):
                court = bounce[
                    "court_position"
                ]

                print(
                    (
                        f"Frame "
                        f"{bounce['frame_number']} "
                        f"| "
                        f"{bounce['timestamp_seconds']}s "
                        f"| "
                        f"{bounce['line_call']} "
                        f"| "
                        f"({court['x_meters']:.2f}, "
                        f"{court['y_meters']:.2f}) m"
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
            "Bounce court analysis successful."
        )

    except (
        VideoLoaderError,
        BallTrackingError,
        CourtHomographyError,
        BounceCourtAnalysisError,
        ValueError,
    ) as error:

        print(
            f"\nBounce analysis failed: "
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