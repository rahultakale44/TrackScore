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
    BallTrajectoryAnalysisError,
    BallTrajectoryAnalyzer,
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
            "Track tennis-ball trajectory and detect "
            "candidate bounce events."
        )
    )

    parser.add_argument(
        "video_path",
        type=str,
        help="Input tennis match video.",
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
            "outputs/ball_trajectory/"
            "ball_trajectory.mp4"
        ),
    )

    parser.add_argument(
        "--metadata",
        type=str,
        default=(
            "outputs/ball_trajectory/"
            "ball_trajectory.json"
        ),
    )

    return parser.parse_args()


def draw_bounce_candidates(
    frame,
    bounce_candidates,
):
    overlay = frame.copy()

    for bounce in bounce_candidates:
        position = bounce[
            "position"
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

        score = float(
            bounce[
                "bounce_score"
            ]
        )

        cv2.circle(
            overlay,
            (
                x,
                y,
            ),
            14,
            (
                0,
                0,
                255,
            ),
            3,
        )

        cv2.putText(
            overlay,
            (
                f"BOUNCE? "
                f"{score:.2f}"
            ),
            (
                x + 18,
                max(
                    y - 10,
                    25,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (
                0,
                0,
                255,
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

        tracker = BallTracker(
            BallTrackerConfig(
                model_path=args.model
            )
        )

        analyzer = (
            BallTrajectoryAnalyzer()
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

            raise BallTrajectoryAnalysisError(
                "Unable to create output video."
            )

        frame_number = 0

        frame_results = []

        global_bounces = []

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
                analyzer.analyse_trajectory(
                    tracker.history
                )
            )

            current_candidates = (
                trajectory_result[
                    "bounce_candidates"
                ]
            )

            global_bounces.extend(
                current_candidates
            )

            deduplicated = (
                analyzer
                .deduplicate_bounces(
                    global_bounces
                )
            )

            overlay = (
                tracker.draw_tracking(
                    frame,
                    tracking_result,
                )
            )

            overlay = (
                draw_bounce_candidates(
                    overlay,
                    deduplicated,
                )
            )

            cv2.putText(
                overlay,
                (
                    f"Bounce Candidates: "
                    f"{len(deduplicated)}"
                ),
                (
                    25,
                    110,
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

                    "track_active": (
                        tracking_result[
                            "track_active"
                        ]
                    ),

                    "bounce_candidates": (
                        current_candidates
                    ),
                }
            )

            frame_number += 1

        capture.release()

        writer.release()

        deduplicated_bounces = (
            analyzer
            .deduplicate_bounces(
                global_bounces
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
            "video": metadata,

            "processed_frames": (
                frame_number
            ),

            "trajectory": (
                tracker.history
            ),

            "bounce_candidate_count": (
                len(
                    deduplicated_bounces
                )
            ),

            "bounce_candidates": (
                deduplicated_bounces
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
            "\nTrackScore Ball Trajectory Analysis"
        )

        print("=" * 65)

        print(
            f"Processed frames: "
            f"{frame_number}"
        )

        print(
            f"Trajectory points: "
            f"{len(tracker.history)}"
        )

        print(
            f"Bounce candidates: "
            f"{len(deduplicated_bounces)}"
        )

        if deduplicated_bounces:
            print(
                "\nDetected Candidates:"
            )

            for bounce in (
                deduplicated_bounces
            ):
                print(
                    (
                        f"Frame "
                        f"{bounce['frame_number']} "
                        f"| "
                        f"{bounce['timestamp_seconds']}s "
                        f"| Score "
                        f"{bounce['bounce_score']}"
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
            "Ball trajectory analysis successful."
        )

    except (
        VideoLoaderError,
        BallTrackingError,
        BallTrajectoryAnalysisError,
        ValueError,
    ) as error:

        print(
            f"\nTrajectory analysis failed: "
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