"""
Tests for video renderer.
"""

import numpy as np
import pytest

from backend.app.vision.video_renderer import (
    VideoRenderer,
    RendererConfig,
    VideoRendererError,
)


def test_renderer_creation():
    """Test renderer can be created."""
    renderer = VideoRenderer()
    assert renderer is not None
    assert renderer.config is not None


def test_renderer_with_custom_config():
    """Test renderer with custom configuration."""
    config = RendererConfig(
        show_scoreboard=False,
        show_ball_markers=True,
        trajectory_length=20,
    )
    renderer = VideoRenderer(config)
    assert not renderer.config.show_scoreboard
    assert renderer.config.show_ball_markers
    assert renderer.config.trajectory_length == 20


def test_calculate_font_scale():
    """Test font scale calculation based on resolution."""
    renderer = VideoRenderer()
    
    # Base resolution (720p)
    scale_720 = renderer._calculate_font_scale(720)
    assert scale_720 == renderer.config.font_scale_base
    
    # Higher resolution (1080p)
    scale_1080 = renderer._calculate_font_scale(1080)
    assert scale_1080 > scale_720
    
    # Lower resolution (480p)
    scale_480 = renderer._calculate_font_scale(480)
    assert scale_480 < scale_720


def test_calculate_font_thickness():
    """Test font thickness calculation based on resolution."""
    renderer = VideoRenderer()
    
    # Base resolution
    thickness_720 = renderer._calculate_font_thickness(720)
    assert thickness_720 == renderer.config.font_thickness_base
    
    # Higher resolution
    thickness_1080 = renderer._calculate_font_thickness(1080)
    assert thickness_1080 >= thickness_720
    
    # Lower resolution
    thickness_480 = renderer._calculate_font_thickness(480)
    assert thickness_480 >= 1  # Minimum thickness


def test_draw_text_with_background():
    """Test text rendering with background."""
    renderer = VideoRenderer()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    renderer._draw_text_with_background(
        frame,
        "Test Text",
        (50, 50),
        0.6,
        2,
        (255, 255, 255),
        (0, 0, 0),
    )
    
    # Check that frame was modified
    assert np.any(frame > 0)


def test_draw_header():
    """Test header rendering."""
    renderer = VideoRenderer()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    renderer._draw_header(frame, 12.5, 0.6, 2)
    
    # Check that frame was modified (header drawn)
    assert np.any(frame > 0)


def test_draw_scoreboard():
    """Test scoreboard rendering."""
    renderer = VideoRenderer()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    scoreboard = {
        "points": {"Player A": "40", "Player B": "30"},
        "games": {"Player A": 3, "Player B": 2},
        "sets": {"Player A": 1, "Player B": 0},
    }
    
    renderer._draw_scoreboard(frame, scoreboard, 0.6, 2)
    
    # Check that frame was modified
    assert np.any(frame > 0)


def test_draw_scoreboard_with_winner():
    """Test scoreboard with winner display."""
    renderer = VideoRenderer()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    scoreboard = {
        "points": {"Player A": "0", "Player B": "0"},
        "games": {"Player A": 6, "Player B": 3},
        "sets": {"Player A": 2, "Player B": 0},
        "winner": "Player A",
    }
    
    renderer._draw_scoreboard(frame, scoreboard, 0.6, 2)
    
    assert np.any(frame > 0)


def test_draw_scoreboard_disabled():
    """Test scoreboard rendering when disabled."""
    config = RendererConfig(show_scoreboard=False)
    renderer = VideoRenderer(config)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    scoreboard = {
        "points": {"Player A": "40", "Player B": "30"},
        "games": {"Player A": 3, "Player B": 2},
        "sets": {"Player A": 1, "Player B": 0},
    }
    
    renderer._draw_scoreboard(frame, scoreboard, 0.6, 2)
    
    # Frame should remain black (nothing drawn)
    assert not np.any(frame > 0)


def test_draw_ball_marker_detected():
    """Test ball marker for detected ball."""
    renderer = VideoRenderer()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    ball = {"x": 100, "y": 200, "predicted": False}
    
    renderer._draw_ball_marker(frame, ball, 0.6)
    
    # Check that marker was drawn
    assert np.any(frame > 0)


def test_draw_ball_marker_predicted():
    """Test ball marker for predicted ball."""
    renderer = VideoRenderer()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    ball = {"x": 100, "y": 200, "predicted": True}
    
    renderer._draw_ball_marker(frame, ball, 0.6)
    
    # Check that marker was drawn
    assert np.any(frame > 0)


def test_draw_trajectory():
    """Test trajectory rendering."""
    renderer = VideoRenderer()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    # Add some trajectory points
    renderer.trajectory_buffer = [
        {"x": 100, "y": 100},
        {"x": 110, "y": 105},
        {"x": 120, "y": 110},
        {"x": 130, "y": 115},
    ]
    
    renderer._draw_trajectory(frame)
    
    # Check that trajectory was drawn
    assert np.any(frame > 0)


def test_trajectory_buffer_limit():
    """Test trajectory buffer respects length limit."""
    config = RendererConfig(trajectory_length=5)
    renderer = VideoRenderer(config)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    # Add more points than limit
    for i in range(10):
        ball = {"x": 100 + i * 10, "y": 100 + i * 5}
        frame_data = {"ball": ball, "timestamp_seconds": i * 0.1}
        renderer.render_frame(frame, frame_data)
    
    # Buffer should be limited
    assert len(renderer.trajectory_buffer) <= config.trajectory_length


def test_draw_player_info():
    """Test player information rendering."""
    renderer = VideoRenderer()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    players = [
        {
            "label": "Player A",
            "bbox": [100, 100, 200, 300],
            "speed_kmh": 15.5,
        },
        {
            "label": "Player B",
            "bbox": [500, 100, 600, 300],
            "speed_kmh": 12.3,
        },
    ]
    
    renderer._draw_player_info(frame, players, 0.6, 2)
    
    # Check that player info was drawn
    assert np.any(frame > 0)


def test_draw_player_info_without_speed():
    """Test player info rendering without speed data."""
    renderer = VideoRenderer()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    players = [
        {
            "label": "Player A",
            "bbox": [100, 100, 200, 300],
        }
    ]
    
    renderer._draw_player_info(frame, players, 0.6, 2)
    
    assert np.any(frame > 0)


def test_draw_event_overlay():
    """Test event overlay rendering."""
    renderer = VideoRenderer()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    event = {
        "type": "BOUNCE",
        "description": "In",
    }
    
    renderer._draw_event_overlay(frame, event, 0.6, 2)
    
    # Check that event was drawn
    assert np.any(frame > 0)


def test_render_frame_complete():
    """Test complete frame rendering with all overlays."""
    renderer = VideoRenderer()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    frame_data = {
        "timestamp_seconds": 5.5,
        "scoreboard": {
            "points": {"Player A": "30", "Player B": "15"},
            "games": {"Player A": 2, "Player B": 1},
            "sets": {"Player A": 0, "Player B": 0},
        },
        "ball": {"x": 640, "y": 360, "predicted": False},
        "players": [
            {"label": "Player A", "bbox": [100, 100, 200, 300], "speed_kmh": 10.5}
        ],
        "events": [{"type": "SHOT", "description": "Forehand"}],
    }
    
    annotated = renderer.render_frame(frame, frame_data)
    
    # Check that annotations were added
    assert annotated.shape == frame.shape
    assert np.any(annotated > 0)


def test_render_frame_minimal_data():
    """Test frame rendering with minimal data."""
    renderer = VideoRenderer()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    frame_data = {}
    
    annotated = renderer.render_frame(frame, frame_data)
    
    # Should still render header
    assert annotated.shape == frame.shape


def test_render_frame_graceful_missing_data():
    """Test graceful handling of missing analytics data."""
    renderer = VideoRenderer()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    frame_data = {
        "timestamp_seconds": 1.0,
        "scoreboard": None,
        "ball": None,
        "players": [],
        "events": [],
    }
    
    annotated = renderer.render_frame(frame, frame_data)
    
    # Should render without crashing
    assert annotated.shape == frame.shape


def test_reset():
    """Test renderer reset."""
    renderer = VideoRenderer()
    
    # Add some trajectory points
    renderer.trajectory_buffer = [
        {"x": 100, "y": 100},
        {"x": 110, "y": 105},
    ]
    
    renderer.reset()
    
    assert len(renderer.trajectory_buffer) == 0


def test_invalid_codec_handling():
    """Test handling of invalid codec."""
    config = RendererConfig(codec="invalid")
    renderer = VideoRenderer(config)
    
    # Should still create renderer (codec validation happens during write)
    assert renderer is not None


def test_render_frame_preserves_original():
    """Test that render_frame doesn't modify original frame."""
    renderer = VideoRenderer()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    original = frame.copy()
    
    frame_data = {
        "ball": {"x": 100, "y": 100, "predicted": False}
    }
    
    annotated = renderer.render_frame(frame, frame_data)
    
    # Original frame should be unchanged
    assert np.array_equal(frame, original)
    # Annotated should be different
    assert not np.array_equal(annotated, original)
