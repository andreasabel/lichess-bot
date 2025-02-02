"""
Some example classes for people who want to create a homemade bot.

With these classes, bot makers will not have to implement the UCI or XBoard interfaces themselves.
"""
from typing import Tuple
import chess
from chess.engine import PlayResult, Limit
import math
import random
from lib.engine_wrapper import MinimalEngine
from lib.lichess_types import MOVE, HOMEMADE_ARGS_TYPE
import logging


# Use this logger variable to print messages to the console or log files.
# logger.info("message") will always print "message" to the console or log file.
# logger.debug("message") will only print "message" if verbose logging is enabled.
logger = logging.getLogger(__name__)

class ExampleEngine(MinimalEngine):
    """An example engine that all homemade engines inherit."""


# Alpha-beta pruning.
class AlphaBeta(ExampleEngine):
    """Explores the game tree exhaustively to a certain depth.

    The leaves are rated 1.0 for a win of the current player, 0.0 for a draw, and -1.0 for a loss.
    An inner node gets the flipped evaluation v_i of each move.
    The evaluation of the node is the maximum value.

    The frequency of picking move i is h_i = exp(k * v_i), where k is a parameter.
    """

    visited = 0

    def search(self, board: chess.Board, *args: HOMEMADE_ARGS_TYPE) -> PlayResult:  # noqa: ARG002
        """Choose a random move according to the distribution generated from iterative evaluation."""
        k = 5.0
        depth = 4

        # Get moves and values for the current board.
        self.visited = 0
        (moves, values) = self.policy(board, depth, -1.0, 1.0)
        logger.info(f"Visited {self.visited} nodes.")
        rated_moves = sorted([(moves[i], values[i]) for i in range(len(moves))], key=lambda x: x[1], reverse=True)
        logger.info(f"Rated moves: {rated_moves}")

        # Calculate the policy.
        weighted = [math.exp(k * v) for v in values]
        # total = sum(weighted)
        # policy = sorted([(moves[i], weighted[i] / total) for i in range(len(moves))], key=lambda x: x[1], reverse=True)
        # # policy = {moves[i]: weighted[i] / total for i in range(len(moves))}
        # logger.info(f"Policy: {policy}")

        # Choose a move.
        [move] = random.choices(population=moves, weights=weighted, k=1)
        logger.info(f"Move: {move}")

        return PlayResult(move, None)

    # Alpha (α) and beta (β) represent lower and upper bounds for child node values at a given tree depth.
    def value(self, board: chess.Board, depth: int, alpha, beta: float) -> float:
        self.visited += 1
        outcome = board.outcome()
        if outcome is not None:
            if outcome.winner is None:
                return 0.0
            if outcome.winner == board.turn:
                return 1.0
            return -1.0
        if depth <= 0:
            return 0.0
        (moves, values) = self.policy(board, depth-1, alpha, beta)
        return max(values)

    def policy(self, board: chess.Board, depth: int, alpha, beta: float) -> Tuple[list[chess.Move], list[float]]:
        moves = list(board.legal_moves)
        # values = [self.value(board.move(m), depth) for m in moves]
        values = [-9.9] * len(moves)
        for i in range(len(moves)):
            board.push(moves[i])
            v = -self.value(board, depth, -beta, -alpha)
            board.pop()
            values[i] = v
            alpha = max(alpha, v)
            if alpha >= beta:
                break
        return (moves, values)



# Classic min-max.
class MinMax(ExampleEngine):
    """Explores the game tree exhaustively to a certain depth.

    The leaves are rated 1.0 for a win of the current player, 0.0 for a draw, and -1.0 for a loss.
    An inner node gets the flipped evaluation v_i of each move.
    The evaluation of the node is the maximum value.

    The frequency of picking move i is h_i = exp(k * v_i), where k is a parameter.
    """

    visited = 0

    def search(self, board: chess.Board, *args: HOMEMADE_ARGS_TYPE) -> PlayResult:  # noqa: ARG002
        """Choose a random move according to the distribution generated from iterative evaluation."""
        k = 5.0
        depth = 3

        # Get moves and values for the current board.
        self.visited = 0
        (moves, values) = self.policy(board, depth)
        logger.info(f"Visited {self.visited} nodes.")
        rated_moves = sorted([(moves[i], values[i]) for i in range(len(moves))], key=lambda x: x[1], reverse=True)
        logger.info(f"Rated moves: {rated_moves}")

        # Calculate the policy.
        weighted = [math.exp(k * v) for v in values]
        # total = sum(weighted)
        # policy = sorted([(moves[i], weighted[i] / total) for i in range(len(moves))], key=lambda x: x[1], reverse=True)
        # # policy = {moves[i]: weighted[i] / total for i in range(len(moves))}
        # logger.info(f"Policy: {policy}")

        # Choose a move.
        [move] = random.choices(population=moves, weights=weighted, k=1)
        logger.info(f"Move: {move}")

        return PlayResult(move, None)


    def value(self, board: chess.Board, depth: int) -> float:
        self.visited += 1
        outcome = board.outcome()
        if outcome is not None:
            if outcome.winner is None:
                return 0.0
            if outcome.winner == board.turn:
                return 1.0
            return -1.0
        if depth <= 0:
            return 0.0
        (moves, values) = self.policy(board, depth-1)
        return max(values)

    def policy(self, board: chess.Board, depth: int) -> Tuple[list[chess.Move], list[float]]:
        moves = list(board.legal_moves)
        # values = [self.value(board.move(m), depth) for m in moves]
        values = [0.0] * len(moves)
        for i in range(len(moves)):
            m = moves[i]
            board.push(m)
            values[i] = -self.value(board, depth)
            board.pop()
        return (moves, values)



# An iterated version of RandomMove.
class IteratedRandomMove(ExampleEngine):
    """Explores the game tree exhaustively to a certain depth.

    The leaves are rated 1.0 for a win of the current player, 0.0 for a draw, and -1.0 for a loss.
    An inner node gets the flipped evaluation v_i of each move.
    The frequency of picking move i is h_i = exp(k * v_i), where k is a parameter.
    The probability of picking move i is h_i / sum(h_i).
    The evaluation of the node is then the expected value, so sum(h_i * v_i) / sum(h_i).
    """

    visited = 0

    def search(self, board: chess.Board, *args: HOMEMADE_ARGS_TYPE) -> PlayResult:  # noqa: ARG002
        """Choose a random move according to the distribution generated from iterative evaluation."""
        k = 5.0
        depth = 3

        self.visited = 0
        (moves, values, weighted, total) = self.policy(board, depth, k)
        logger.info(f"Visited {self.visited} nodes.")

        policy = {moves[i]: weighted[i] / total for i in range(len(moves))}
        logger.info(f"Policy: {policy}")

        [move] = random.choices(population=moves, weights=weighted, k=1)
        logger.info(f"Move: {move}")

        return PlayResult(move, None)


    def value(self, board: chess.Board, depth: int, k: float) -> float:
        self.visited += 1
        outcome = board.outcome()
        if outcome is not None:
            if outcome.winner is None:
                return 0.0
            if outcome.winner == board.turn:
                return 1.0
            return -1.0
        if depth <= 0:
            return 0.0
        (moves, values, weighted, total) = self.policy(board, depth-1, k)
        v = sum(weighted[i] * values[i] for i in range(len(moves))) / total
        return v

    def policy(self, board: chess.Board, depth: int, k: float) -> Tuple[list[chess.Move], list[float], list[float], float]:
        moves = list(board.legal_moves)
        # values = [self.value(board.move(m), depth, k) for m in moves]
        values = [0.0] * len(moves)
        for i in range(len(moves)):
            m = moves[i]
            board.push(m)
            values[i] = self.value(board, depth, k)  ## BUG: should be modified since player changes
            board.pop()
        weighted = [math.exp(k * v) for v in values]
        total = sum(weighted)
        return (moves, values, weighted, total)

# Bot names and ideas from tom7's excellent eloWorld video

class RandomMove(ExampleEngine):
    """Get a random move."""

    def search(self, board: chess.Board, *args: HOMEMADE_ARGS_TYPE) -> PlayResult:  # noqa: ARG002
        """Choose a random move."""
        return PlayResult(random.choice(list(board.legal_moves)), None)


class Alphabetical(ExampleEngine):
    """Get the first move when sorted by san representation."""

    def search(self, board: chess.Board, *args: HOMEMADE_ARGS_TYPE) -> PlayResult:  # noqa: ARG002
        """Choose the first move alphabetically."""
        moves = list(board.legal_moves)
        moves.sort(key=board.san)
        return PlayResult(moves[0], None)


class FirstMove(ExampleEngine):
    """Get the first move when sorted by uci representation."""

    def search(self, board: chess.Board, *args: HOMEMADE_ARGS_TYPE) -> PlayResult:  # noqa: ARG002
        """Choose the first move alphabetically in uci representation."""
        moves = list(board.legal_moves)
        moves.sort(key=str)
        return PlayResult(moves[0], None)


class ComboEngine(ExampleEngine):
    """
    Get a move using multiple different methods.

    This engine demonstrates how one can use `time_limit`, `draw_offered`, and `root_moves`.
    """

    def search(self,
               board: chess.Board,
               time_limit: Limit,
               ponder: bool,  # noqa: ARG002
               draw_offered: bool,
               root_moves: MOVE) -> PlayResult:
        """
        Choose a move using multiple different methods.

        :param board: The current position.
        :param time_limit: Conditions for how long the engine can search (e.g. we have 10 seconds and search up to depth 10).
        :param ponder: Whether the engine can ponder after playing a move.
        :param draw_offered: Whether the bot was offered a draw.
        :param root_moves: If it is a list, the engine should only play a move that is in `root_moves`.
        :return: The move to play.
        """
        if isinstance(time_limit.time, int):
            my_time = time_limit.time
            my_inc = 0
        elif board.turn == chess.WHITE:
            my_time = time_limit.white_clock if isinstance(time_limit.white_clock, int) else 0
            my_inc = time_limit.white_inc if isinstance(time_limit.white_inc, int) else 0
        else:
            my_time = time_limit.black_clock if isinstance(time_limit.black_clock, int) else 0
            my_inc = time_limit.black_inc if isinstance(time_limit.black_inc, int) else 0

        possible_moves = root_moves if isinstance(root_moves, list) else list(board.legal_moves)

        if my_time / 60 + my_inc > 10:
            # Choose a random move.
            move = random.choice(possible_moves)
        else:
            # Choose the first move alphabetically in uci representation.
            possible_moves.sort(key=str)
            move = possible_moves[0]
        return PlayResult(move, None, draw_offered=draw_offered)
