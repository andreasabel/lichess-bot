"""
This module contains the base class for a player in a chess game.

The player is responsible for picking a move given a board.
The player can also simulate a self-play game (rollout) from the current position.

This module also defines some simple players:
- Player: returns the first legal move.
- RandomPlayer: returns a random legal move.
- GreedyPlayer: returns a move that directly wins, or a move that does not directly lose.

"""

import random
import chess

# Exception a player can raise to abort the search when time is out.

class TimeOutException(Exception):
    """Exception raised for a timeout."""
    def __init__(self, message : str = "Operation timed out"):
        self.message = message
        super().__init__(self.message)

# List of moves with a rating to be used in players.

class RatedMoveSequence:
    """A sequence of moves with rating of the sequence."""

    def __init__(self, rating: float, moves: list[chess.Move] = []):
        self.moves = moves
        self.rating = rating

    def __str__(self):
        return f"RatedMoveSequence({self.rating}, {self.moves})"

    def __repr__(self):
        return str(self)

# Base player: returns the first legal move.

class Player:
    """A player picking a move given a board."""

    name = "FirstMove" # Name of the player.

    def __init__(self):
        # Free-form information about the move computed by the move() method.
        self.move_info = ""

    def move(self, board: chess.Board) -> chess.Move:
        return next(board.generate_legal_moves())

    def rollout(self, board: chess.Board, max_depth: int) -> float:
        """Simulate a game from the current position, destroying the board.
        The returned value is from the perspective of WHITE.
        So it is 1.0 if WHITE wins, 0.0 if it is a draw, and -1.0 if BLACK wins."""
        outcome = board.outcome()
        depth = 0
        while outcome is None and depth < max_depth:
            board.push(self.move(board))
            outcome = board.outcome()
            depth += 1
        if outcome is None or outcome.winner is None:
            return 0.0
        if outcome.winner: # WHITE wins
            return 1.0
        return -1.0


class RandomPlayer(Player):
    """A player picking a random move."""

    name = "Random" # Name of the player.

    def move(self, board: chess.Board) -> chess.Move:
        return random.choice(list(board.legal_moves))


class GreedyPlayer(Player):
    """A player picking a move that directly wins, or a move that does not directly lose."""

    name = "Greedy" # Name of the player.

    def __init__(self):
        super().__init__()
        # Number of positions visited during the search.
        self.visited = 0

    def move(self, board: chess.Board) -> chess.Move:
        """Choose a move that directly wins, or a move that does not directly lose.
        Restores the board state."""
        non_losing_moves: list[chess.Move] = []
        moves = list(board.legal_moves)
        for m in moves:
            self.visited += 1
            board.push(m)
            outcome = board.outcome()
            board.pop()
            # Search for a non-losing move.
            if outcome is None or outcome.winner is None:
                non_losing_moves.append(m)
            # If we find a winning move, commit to it and stop the search.
            elif outcome.winner == board.turn:
                return m
        # If we did not find a winning move, randomly choose a non-losing move.
        if non_losing_moves:
            return random.choice(non_losing_moves)
        # If there are no only losing moves, return one of then randomly.
        return random.choice(moves)
