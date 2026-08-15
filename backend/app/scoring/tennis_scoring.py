from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional


class TennisScoringError(Exception):
    """Raised when tennis scoring state is invalid."""


@dataclass
class TennisScoringConfig:
    """
    Configuration for tennis scoring.
    """

    player_a_name: str = "Player A"
    player_b_name: str = "Player B"

    games_to_win_set: int = 6
    minimum_game_difference: int = 2

    enable_tiebreak: bool = True
    tiebreak_trigger_games: int = 6
    tiebreak_points_to_win: int = 7
    tiebreak_minimum_difference: int = 2

    sets_to_win_match: int = 2


class TennisScoringEngine:
    """
    Full tennis scoring state engine.

    Supports:
    - 0 / 15 / 30 / 40
    - Deuce
    - Advantage
    - Game win
    - Set win
    - Tie-break
    - Match winner
    - Score history
    """

    POINT_LABELS = {
        0: "0",
        1: "15",
        2: "30",
        3: "40",
    }

    def __init__(
        self,
        config: TennisScoringConfig | None = None,
    ):
        self.config = (
            config
            if config is not None
            else TennisScoringConfig()
        )

        self._validate_config()

        self._event_listeners: Dict[
            str,
            List[Callable[[Dict[str, Any]], None]],
        ] = defaultdict(list)
        self.event_listeners = self._event_listeners

        self.reset_match()

    # ============================================================
    # CONFIG VALIDATION
    # ============================================================

    def _validate_config(self) -> None:
        config = self.config

        if not config.player_a_name:
            raise TennisScoringError(
                "player_a_name cannot be empty."
            )

        if not config.player_b_name:
            raise TennisScoringError(
                "player_b_name cannot be empty."
            )

        if config.games_to_win_set <= 0:
            raise TennisScoringError(
                "games_to_win_set must be greater than zero."
            )

        if config.minimum_game_difference <= 0:
            raise TennisScoringError(
                "minimum_game_difference must be greater than zero."
            )

        if config.tiebreak_points_to_win <= 0:
            raise TennisScoringError(
                "tiebreak_points_to_win must be greater than zero."
            )

        if config.tiebreak_minimum_difference <= 0:
            raise TennisScoringError(
                "tiebreak_minimum_difference must be greater than zero."
            )

        if config.sets_to_win_match <= 0:
            raise TennisScoringError(
                "sets_to_win_match must be greater than zero."
            )

    # ============================================================
    # RESET
    # ============================================================

    def reset_game(self) -> None:
        self.game_points = {
            "Player A": 0,
            "Player B": 0,
        }

        self.advantage_player: Optional[str] = None

    def reset_set(self) -> None:
        self.games = {
            "Player A": 0,
            "Player B": 0,
        }

        self.tiebreak_points = {
            "Player A": 0,
            "Player B": 0,
        }

        self.in_tiebreak = False

        self.reset_game()

    def reset_match(self) -> None:
        self.sets = {
            "Player A": 0,
            "Player B": 0,
        }

        self.completed_sets: List[
            Dict[str, Any]
        ] = []

        self.point_history: List[
            Dict[str, Any]
        ] = []

        self.game_history: List[
            Dict[str, Any]
        ] = []

        self.match_winner: Optional[str] = None

        self.reset_set()

    # ============================================================
    # PLAYER VALIDATION
    # ============================================================

    @staticmethod
    def _validate_player(
        player: str,
    ) -> None:
        if player not in {
            "Player A",
            "Player B",
        }:
            raise TennisScoringError(
                "Player must be 'Player A' or 'Player B'."
            )

    @staticmethod
    def get_opponent(
        player: str,
    ) -> str:
        TennisScoringEngine._validate_player(
            player
        )

        return (
            "Player B"
            if player == "Player A"
            else "Player A"
        )

    def add_event_listener(
        self,
        event_type: str,
        callback: Callable[[Dict[str, Any]], None],
    ) -> None:
        """
        Register a callback for a scoring lifecycle event.
        """

        if not isinstance(
            event_type,
            str,
        ) or not event_type.strip():
            raise TennisScoringError(
                "event_type must be a non-empty string."
            )

        if not callable(callback):
            raise TennisScoringError(
                "callback must be callable."
            )

        self._event_listeners[
            event_type
        ].append(callback)

    def remove_event_listener(
        self,
        event_type: str,
        callback: Callable[[Dict[str, Any]], None],
    ) -> None:
        """
        Remove a registered callback for an event type.
        """

        if event_type not in self._event_listeners:
            return

        self._event_listeners[event_type] = [
            listener
            for listener in self._event_listeners[event_type]
            if listener is not callback
        ]

    def _emit_event(
        self,
        event_type: str,
        **payload: Any,
    ) -> None:
        state = self.get_state()
        event = {
            "type": event_type,
            "timestamp": (
                datetime.now(timezone.utc)
                .isoformat()
            ),
            "state": state,
            **payload,
        }

        for event_name in (
            event_type,
            "*",
        ):
            for listener in list(
                self._event_listeners.get(
                    event_name,
                    [],
                )
            ):
                listener(event)

    def get_live_scoreboard(
        self,
    ) -> Dict[str, Any]:
        """
        Return the latest scoreboard snapshot for UI consumers.
        """

        return self.get_state()

    # ============================================================
    # GAME SCORE DISPLAY
    # ============================================================

    def get_game_score_display(
        self,
    ) -> Dict[str, str]:
        """
        Return traditional tennis point display.
        """

        a = self.game_points[
            "Player A"
        ]

        b = self.game_points[
            "Player B"
        ]

        if (
            a >= 3
            and b >= 3
        ):
            if a == b:
                return {
                    "Player A": "40",
                    "Player B": "40",
                    "status": "DEUCE",
                }

            if a > b:
                return {
                    "Player A": "AD",
                    "Player B": "40",
                    "status": "ADVANTAGE Player A",
                }

            return {
                "Player A": "40",
                "Player B": "AD",
                "status": "ADVANTAGE Player B",
            }

        return {
            "Player A": self.POINT_LABELS.get(
                a,
                "40",
            ),
            "Player B": self.POINT_LABELS.get(
                b,
                "40",
            ),
            "status": "NORMAL",
        }

    # ============================================================
    # POINT WIN
    # ============================================================

    def award_point(
        self,
        player: str,
        metadata: Optional[
            Dict[str, Any]
        ] = None,
    ) -> Dict[str, Any]:
        """
        Award one tennis point.
        """

        self._validate_player(
            player
        )

        if self.match_winner is not None:
            raise TennisScoringError(
                "Match has already finished."
            )

        if self.in_tiebreak:
            return self._award_tiebreak_point(
                player,
                metadata,
            )

        opponent = self.get_opponent(
            player
        )

        self.game_points[
            player
        ] += 1

        game_winner = None

        player_points = self.game_points[
            player
        ]

        opponent_points = self.game_points[
            opponent
        ]

        if (
            player_points >= 4
            and
            player_points - opponent_points >= 2
        ):
            game_winner = player

        self.point_history.append(
            {
                "point_number": len(
                    self.point_history
                ) + 1,
                "winner": player,
                "metadata": metadata or {},
                "game_points": dict(
                    self.game_points
                ),
                "games": dict(
                    self.games
                ),
                "sets": dict(
                    self.sets
                ),
            }
        )

        if game_winner is not None:
            self._award_game(
                game_winner
            )

        self._emit_event(
            "point",
            player=player,
            winner=player,
            metadata=metadata or {},
            game_points=dict(
                self.game_points
            ),
            games=dict(
                self.games
            ),
            sets=dict(
                self.sets
            ),
        )

        return self.get_state()

    # ============================================================
    # GAME WIN
    # ============================================================

    def _award_game(
        self,
        player: str,
    ) -> None:
        self.games[
            player
        ] += 1

        self.game_history.append(
            {
                "game_number": len(
                    self.game_history
                ) + 1,
                "winner": player,
                "games": dict(
                    self.games
                ),
                "sets": dict(
                    self.sets
                ),
            }
        )

        self.reset_game()

        if self._should_start_tiebreak():
            self.in_tiebreak = True
            self._emit_event(
                "game",
                player=player,
                winner=player,
                game_number=len(self.game_history),
                games=dict(self.games),
                sets=dict(self.sets),
            )
            return

        self._emit_event(
            "game",
            player=player,
            winner=player,
            game_number=len(self.game_history),
            games=dict(self.games),
            sets=dict(self.sets),
        )

        if self._has_won_set(
            player
        ):
            self._award_set(
                player
            )

    # ============================================================
    # SET
    # ============================================================

    def _has_won_set(
        self,
        player: str,
    ) -> bool:
        opponent = self.get_opponent(
            player
        )

        player_games = self.games[
            player
        ]

        opponent_games = self.games[
            opponent
        ]

        return (
            player_games
            >= self.config.games_to_win_set
            and
            player_games
            - opponent_games
            >= self.config.minimum_game_difference
        )

    def _award_set(
        self,
        player: str,
    ) -> None:
        opponent = self.get_opponent(
            player
        )

        set_record = {
            "winner": player,

            "Player A": self.games[
                "Player A"
            ],

            "Player B": self.games[
                "Player B"
            ],
        }

        self.completed_sets.append(
            set_record
        )

        self.sets[
            player
        ] += 1

        if (
            self.sets[player]
            >= self.config.sets_to_win_match
        ):
            self.match_winner = player

        self._emit_event(
            "set",
            player=player,
            winner=player,
            set_record=dict(set_record),
            games=dict(self.games),
            sets=dict(self.sets),
        )

        if self.match_winner is not None:
            self._emit_event(
                "match",
                player=player,
                winner=player,
                games=dict(self.games),
                sets=dict(self.sets),
            )

        if self.match_winner is None:
            self.reset_set()

    # ============================================================
    # TIEBREAK
    # ============================================================

    def _should_start_tiebreak(
        self,
    ) -> bool:
        if not self.config.enable_tiebreak:
            return False

        trigger = (
            self.config
            .tiebreak_trigger_games
        )

        return (
            self.games[
                "Player A"
            ] == trigger
            and
            self.games[
                "Player B"
            ] == trigger
        )

    def _award_tiebreak_point(
        self,
        player: str,
        metadata: Optional[
            Dict[str, Any]
        ] = None,
    ) -> Dict[str, Any]:
        opponent = self.get_opponent(
            player
        )

        self.tiebreak_points[
            player
        ] += 1

        self.point_history.append(
            {
                "point_number": len(
                    self.point_history
                ) + 1,

                "winner": player,

                "tiebreak": True,

                "metadata": (
                    metadata or {}
                ),

                "tiebreak_points": dict(
                    self.tiebreak_points
                ),
            }
        )

        player_points = (
            self.tiebreak_points[
                player
            ]
        )

        opponent_points = (
            self.tiebreak_points[
                opponent
            ]
        )

        if (
            player_points
            >= self.config
            .tiebreak_points_to_win
            and
            player_points - opponent_points
            >= self.config
            .tiebreak_minimum_difference
        ):
            self.games[
                player
            ] += 1

            self.in_tiebreak = False

            self._emit_event(
                "point",
                player=player,
                winner=player,
                metadata=metadata or {},
                tiebreak=True,
                tiebreak_points=dict(
                    self.tiebreak_points
                ),
                games=dict(self.games),
                sets=dict(self.sets),
            )

            self._award_set(
                player
            )

        return self.get_state()

    # ============================================================
    # LEADER
    # ============================================================

    def get_current_leader(
        self,
    ) -> str:
        if self.match_winner:
            return self.match_winner

        if (
            self.sets["Player A"]
            >
            self.sets["Player B"]
        ):
            return "Player A"

        if (
            self.sets["Player B"]
            >
            self.sets["Player A"]
        ):
            return "Player B"

        if (
            self.games["Player A"]
            >
            self.games["Player B"]
        ):
            return "Player A"

        if (
            self.games["Player B"]
            >
            self.games["Player A"]
        ):
            return "Player B"

        a = self.game_points[
            "Player A"
        ]

        b = self.game_points[
            "Player B"
        ]

        if a > b:
            return "Player A"

        if b > a:
            return "Player B"

        return "TIED"

    # ============================================================
    # STATE
    # ============================================================

    def get_state(
        self,
    ) -> Dict[str, Any]:
        point_display = (
            self.get_game_score_display()
            if not self.in_tiebreak
            else {
                "Player A": str(
                    self.tiebreak_points[
                        "Player A"
                    ]
                ),
                "Player B": str(
                    self.tiebreak_points[
                        "Player B"
                    ]
                ),
                "status": "TIEBREAK",
            }
        )

        return {
            "players": {
                "Player A": (
                    self.config.player_a_name
                ),
                "Player B": (
                    self.config.player_b_name
                ),
            },

            "points": point_display,

            "games": dict(
                self.games
            ),

            "sets": dict(
                self.sets
            ),

            "completed_sets": list(
                self.completed_sets
            ),

            "in_tiebreak": (
                self.in_tiebreak
            ),

            "tiebreak_points": dict(
                self.tiebreak_points
            ),

            "leader": (
                self.get_current_leader()
            ),

            "match_winner": (
                self.match_winner
            ),

            "match_finished": (
                self.match_winner
                is not None
            ),
        }