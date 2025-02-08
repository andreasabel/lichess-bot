
import random
from typing import Generator
import chess
from logging import Logger

from engines.player import Player, RatedMoveSequence, TimeOutException


class AlphaBetaPlayer(Player):
    """A player using Alpha-Beta pruning."""

    def __init__(self, depth: int = 3, decay: float = 1.0):
        self.depth = depth
        self.decay = decay

        # Count the number of visited positions in a search.
        self.visited = 0

        # Data computed by the move() method:
        #####################################

        # Policy is a list of RatedMoveSequence.
        self.move_sequences: list[RatedMoveSequence] = []

        # Best rating found.
        self.best_rating = 0.0
        self.best_move_sequences: list[list[chess.Move]] = []

        # Randomly chosen move from the best moves.
        self.chosen_move: chess.Move | None = None

    def reset_visited(self):
        """Reset the number of visited nodes."""
        self.visited = 0

    def inc_visited(self):
        """Increment the number of visited nodes.
        This method can be overridden to act if
        the number of visits exceeds a certain threshold."""
        self.visited += 1


    def move(self, board: chess.Board) -> chess.Move:
        """Choose a move determined by complete exploration up to self.depth optimized by alpha-beta pruning.
        This method sets the following attributes:
        - self.visited
        - self.move_sequences
        - self.best_rating
        - self.best_move_sequences
        - self.chosen_move
        """

        # Reset the number of visited nodes.
        self.reset_visited()

        # Get moves and values for the current board.
        self.move_sequences = list(self.policy(board, self.depth))

        # Find the best rating.
        self.best_rating = max(self.move_sequences, key=lambda x: x.rating).rating

        # Find the best moves.
        self.best_move_sequences = [rms.moves for rms in self.move_sequences if rms.rating >= self.best_rating]

        # Choose one of the best moves.
        moves = [ms[0] for ms in self.best_move_sequences]
        self.chosen_move = random.choice(moves)
        return self.chosen_move

    def log(self, logger: Logger, board: chess.Board):
        """Print the search results and statistics of the last move() invocation."""

        logger.info(f"Visited {self.visited} nodes.")

        logger.info(f"Best rating: {self.best_rating}")
        logger.info(f"Best move sequences:")
        for ms in self.best_move_sequences:
            logger.info(f"- {board.variation_san(ms)}")

        assert self.chosen_move is not None
        logger.info(f"Move: {board.lan(self.chosen_move)}")

    def policy(self, board: chess.Board, depth: int) -> Generator[RatedMoveSequence]:
        """Compute the value of each legal move of the current board by exploring the game tree up to the given cdepth."""
        for m in board.legal_moves:
            board.push(m)
            rms = self.value(board, depth, -1.0, 1.0)
            board.pop()
            yield RatedMoveSequence(-rms.rating, [m] + rms.moves)


    # Alpha (α) and beta (β) represent lower and upper bounds for child node values at a given tree depth.
    def value(self, board: chess.Board, depth: int, alpha: float, beta: float) -> RatedMoveSequence:
        """Compute the value of the current board by exploring the game tree up to the given certain depth."""
        self.inc_visited()

        # If game is finished or depth exhausted, give the immediate evaluation.
        outcome = board.outcome()
        if outcome is not None:
            if outcome.winner is None:
                return RatedMoveSequence(0.0)
            if outcome.winner == board.turn:
                return RatedMoveSequence(1.0)
            return RatedMoveSequence(-1.0)
        if depth <= 0:
            return RatedMoveSequence(0.0)

        # Otherwise, evaluate the child nodes.
        best_move = None
        for m in board.legal_moves:
            board.push(m)
            rms = self.value(board, depth-1, -beta, -alpha)
            board.pop()
            v = -self.decay * rms.rating
            if v > alpha:
                best_move = None
                alpha = v
            if best_move is None:
                best_move = RatedMoveSequence(v, [m] + rms.moves)
            if alpha >= beta:
                break
        return best_move # type: ignore

class IteratedDeepeningAlphaBetaPlayer(AlphaBetaPlayer):
    """A player using Iterated Deepening Alpha-Beta pruning."""

    def __init__(self, max_visits: int = 1000000, round_start_limit: int = 80000, decay: float = 1.0):

        # Initialize AlphaBetaPlayer with depth=0, to be incremented in each round.
        super().__init__(depth=0, decay=decay)

        # The total number of positions we are allowed to visit.
        self.max_visits = max_visits

        # The max numer of visits we can have spent before starting a new round.
        self.round_start_limit = round_start_limit

    def reset_visited(self):
        """Skip the reset."""
        pass

    def inc_visited(self):  # type: ignore
        """Increment the number of visited nodes.
        If the number of visits exceeds max_visits, throw a TimeOutException."""
        self.visited += 1
        if self.visited > self.max_visits:
            raise TimeOutException()

    def move(self, board: chess.Board) -> chess.Move:
        """Choose a move determined by iterative deepening optimized by alpha-beta pruning."""

        self.visited = 0
        best_possible_rating = self.decay

        for self.depth in range(1, 100):
            if self.visited > self.round_start_limit:
                break
            try:
                # Initialize best_rating, best_move_sequences, and chosen_move.
                super().move(board.copy())
            except TimeOutException:
                self.depth -= 1
                break

            if self.best_rating >= best_possible_rating or self.best_rating <= -best_possible_rating:
                break
            best_possible_rating = best_possible_rating * self.decay

        assert self.chosen_move is not None
        return self.chosen_move
