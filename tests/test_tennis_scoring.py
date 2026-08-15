import pytest

from backend.app.scoring.tennis_scoring import (
    TennisScoringEngine,
    TennisScoringError,
)


def win_game(
    engine,
    player,
):
    for _ in range(
        4
    ):
        engine.award_point(
            player
        )


def test_initial_score():
    engine = (
        TennisScoringEngine()
    )

    state = (
        engine.get_state()
    )

    assert (
        state[
            "points"
        ][
            "Player A"
        ]
        == "0"
    )

    assert (
        state[
            "games"
        ][
            "Player A"
        ]
        == 0
    )


def test_first_point():
    engine = (
        TennisScoringEngine()
    )

    engine.award_point(
        "Player A"
    )

    state = (
        engine.get_state()
    )

    assert (
        state[
            "points"
        ][
            "Player A"
        ]
        == "15"
    )


def test_30_score():
    engine = (
        TennisScoringEngine()
    )

    engine.award_point(
        "Player A"
    )

    engine.award_point(
        "Player A"
    )

    assert (
        engine
        .get_state()[
            "points"
        ][
            "Player A"
        ]
        == "30"
    )


def test_40_score():
    engine = (
        TennisScoringEngine()
    )

    for _ in range(
        3
    ):
        engine.award_point(
            "Player A"
        )

    assert (
        engine
        .get_state()[
            "points"
        ][
            "Player A"
        ]
        == "40"
    )


def test_game_win():
    engine = (
        TennisScoringEngine()
    )

    win_game(
        engine,
        "Player A",
    )

    state = (
        engine.get_state()
    )

    assert (
        state[
            "games"
        ][
            "Player A"
        ]
        == 1
    )

    assert (
        state[
            "points"
        ][
            "Player A"
        ]
        == "0"
    )


def test_deuce():
    engine = (
        TennisScoringEngine()
    )

    for _ in range(
        3
    ):
        engine.award_point(
            "Player A"
        )

        engine.award_point(
            "Player B"
        )

    state = (
        engine.get_state()
    )

    assert (
        state[
            "points"
        ][
            "status"
        ]
        == "DEUCE"
    )


def test_advantage():
    engine = (
        TennisScoringEngine()
    )

    for _ in range(
        3
    ):
        engine.award_point(
            "Player A"
        )

        engine.award_point(
            "Player B"
        )

    engine.award_point(
        "Player A"
    )

    state = (
        engine.get_state()
    )

    assert (
        state[
            "points"
        ][
            "Player A"
        ]
        == "AD"
    )


def test_advantage_back_to_deuce():
    engine = (
        TennisScoringEngine()
    )

    for _ in range(
        3
    ):
        engine.award_point(
            "Player A"
        )

        engine.award_point(
            "Player B"
        )

    engine.award_point(
        "Player A"
    )

    engine.award_point(
        "Player B"
    )

    assert (
        engine
        .get_state()[
            "points"
        ][
            "status"
        ]
        == "DEUCE"
    )


def test_set_win():
    engine = (
        TennisScoringEngine()
    )

    for _ in range(
        6
    ):
        win_game(
            engine,
            "Player A",
        )

    assert (
        engine
        .get_state()[
            "sets"
        ][
            "Player A"
        ]
        == 1
    )


def test_match_win():
    engine = (
        TennisScoringEngine()
    )

    for _ in range(
        2
    ):
        for _ in range(
            6
        ):
            win_game(
                engine,
                "Player A",
            )

    state = (
        engine.get_state()
    )

    assert (
        state[
            "match_winner"
        ]
        == "Player A"
    )

    assert (
        state[
            "match_finished"
        ]
        is True
    )


def test_invalid_player():
    engine = (
        TennisScoringEngine()
    )

    with pytest.raises(
        TennisScoringError
    ):
        engine.award_point(
            "Player C"
        )


def test_leader():
    engine = (
        TennisScoringEngine()
    )

    engine.award_point(
        "Player B"
    )

    assert (
        engine.get_current_leader()
        == "Player B"
    )


def test_point_history():
    engine = (
        TennisScoringEngine()
    )

    engine.award_point(
        "Player A",
        metadata={
            "reason": "winner"
        },
    )

    assert (
        len(
            engine.point_history
        )
        == 1
    )

    assert (
        engine
        .point_history[
            0
        ][
            "metadata"
        ][
            "reason"
        ]
        == "winner"
    )