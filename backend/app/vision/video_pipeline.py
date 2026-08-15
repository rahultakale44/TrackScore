from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type


class VideoPipelineError(Exception):
    """Raised when the unified video analytics pipeline fails."""


@dataclass
class VideoPipelineConfig:
    """Configuration for a unified tennis video analytics pipeline."""

    interval: int = 1
    max_frames: Optional[int] = None
    output_dir: str = "outputs/video_pipeline"
    target_width: int = 640
    target_height: int = 640


class VideoAnalyticsPipeline:
    """
    Unified analytics pipeline that sequences the existing video-processing modules.

    Responsibilities:
    - validate source video
    - extract frames
    - preprocess frames
    - run detector passes for players and ball
    - package a unified result object for downstream analytics
    """

    def __init__(
        self,
        video_path: str,
        output_dir: str,
        video_loader_cls: Optional[Type[Any]] = None,
        frame_extractor_cls: Optional[Type[Any]] = None,
        preprocessor_cls: Optional[Type[Any]] = None,
        player_detector_cls: Optional[Type[Any]] = None,
        ball_detector_cls: Optional[Type[Any]] = None,
        config: Optional[VideoPipelineConfig] = None,
    ):
        self.video_path = str(video_path)
        self.output_dir = str(output_dir)
        self.config = config if config is not None else VideoPipelineConfig()

        self.video_loader_cls = video_loader_cls or self._default_video_loader
        self.frame_extractor_cls = frame_extractor_cls or self._default_frame_extractor
        self.preprocessor_cls = preprocessor_cls or self._default_preprocessor
        self.player_detector_cls = player_detector_cls or self._default_player_detector
        self.ball_detector_cls = ball_detector_cls or self._default_ball_detector

    @staticmethod
    def _default_video_loader(video_path: str):
        from backend.app.vision.video_loader import VideoLoader

        return VideoLoader(video_path)

    @staticmethod
    def _default_frame_extractor(video_path: str, output_dir: str):
        from backend.app.vision.frame_extractor import FrameExtractor

        return FrameExtractor(video_path, output_dir)

    @staticmethod
    def _default_preprocessor(target_width: int = 640, target_height: int = 640, quality_config=None):
        from backend.app.vision.frame_preprocessor import FramePreprocessor

        return FramePreprocessor(
            target_width=target_width,
            target_height=target_height,
            quality_config=quality_config,
        )

    @staticmethod
    def _default_player_detector(config=None):
        from backend.app.vision.player_detector import PlayerDetector

        return PlayerDetector(config)

    @staticmethod
    def _default_ball_detector(config=None):
        from backend.app.vision.ball_detector import BallDetector

        return BallDetector(config)

    def _load_video_metadata(self) -> Dict[str, Any]:
        loader = self.video_loader_cls(self.video_path)
        if hasattr(loader, "get_metadata"):
            return loader.get_metadata()
        raise VideoPipelineError("Video loader must provide get_metadata().")

    def _extract_frames(self) -> List[Dict[str, Any]]:
        extractor = self.frame_extractor_cls(
            self.video_path,
            self.output_dir,
        )

        if not hasattr(extractor, "extract_every_n_frames"):
            raise VideoPipelineError(
                "Frame extractor must provide extract_every_n_frames()."
            )

        return extractor.extract_every_n_frames(
            self.config.interval,
            self.config.max_frames,
        )

    def _preprocess_frames(self, frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        preprocessor = self.preprocessor_cls(
            target_width=self.config.target_width,
            target_height=self.config.target_height,
        )

        processed_frames: List[Dict[str, Any]] = []
        for frame_record in frames:
            frame = frame_record.get("frame")
            if frame is None:
                frame = frame_record.get("image")
            if frame is None:
                raise VideoPipelineError("Frame record is missing frame/image data.")

            processed_frame, metadata = preprocessor.process_frame(frame)
            updated_record = dict(frame_record)
            updated_record["processed_frame"] = processed_frame
            updated_record["preprocessing"] = metadata
            processed_frames.append(updated_record)

        return processed_frames

    def _run_players_detection(self, frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        detector = self.player_detector_cls()
        result: List[Dict[str, Any]] = []

        for frame_record in frames:
            frame = frame_record["processed_frame"]
            detection = detector.detect_players(frame)
            result.append({
                "frame_number": frame_record.get("frame_number"),
                "timestamp_seconds": frame_record.get("timestamp_seconds"),
                "players": detection.get("players", []),
            })

        return result

    def _run_ball_detection(self, frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        detector = self.ball_detector_cls()
        result: List[Dict[str, Any]] = []

        for frame_record in frames:
            frame = frame_record["processed_frame"]
            detection = detector.detect_ball(frame)
            result.append({
                "frame_number": frame_record.get("frame_number"),
                "timestamp_seconds": frame_record.get("timestamp_seconds"),
                "ball": detection.get("balls", detection.get("ball", [])),
            })

        return result

    def run(self, max_frames: Optional[int] = None) -> Dict[str, Any]:
        """
        Run the full unified video-analysis pipeline.
        """

        if max_frames is not None:
            self.config.max_frames = max_frames

        metadata = self._load_video_metadata()
        frames = self._extract_frames()
        processed_frames = self._preprocess_frames(frames)
        player_detections = self._run_players_detection(processed_frames)
        ball_detections = self._run_ball_detection(processed_frames)

        player_results = []
        for item in player_detections:
            player_results.extend(item.get("players", []))

        ball_results = []
        for item in ball_detections:
            ball_results.extend(item.get("ball", []))

        result = {
            "video_metadata": metadata,
            "frame_count": len(processed_frames),
            "frames": [
                {
                    "frame_number": item.get("frame_number"),
                    "timestamp_seconds": item.get("timestamp_seconds"),
                    "filename": item.get("filename"),
                    "path": item.get("path"),
                    "preprocessing": item.get("preprocessing"),
                }
                for item in processed_frames
            ],
            "detections": {
                "players": player_results,
                "ball": ball_results,
            },
            "status": "success",
        }

        return result


__all__ = [
    "VideoAnalyticsPipeline",
    "VideoPipelineConfig",
    "VideoPipelineError",
]
