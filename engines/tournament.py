# Play a tournament between two players.

import chess
from chess.variant import AtomicBoard
import logging
from rich.logging import RichHandler

from engines.player import *
from engines.alphabeta import *
from engines.mcts import MCTSPlayer

# Use this logger variable to print messages to the console or log files.
# logger.info("message") will always print "message" to the console or log file.
# logger.debug("message") will only print "message" if verbose logging is enabled.
logger = logging.getLogger(__name__)


def main():

    level = logging.DEBUG

    # Set up pretty logging to the console.
    # Code from lib/lichess-bot.py
    console_handler = RichHandler()
    console_formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(level)
    all_handlers: list[logging.Handler] = [console_handler]
    logging.basicConfig(level=level, handlers=all_handlers, force=True)

    # (wins, losses, draws) = tournament(GreedyPlayer(), RandomPlayer(), games=100)
    # print(f"GreedyPlayer vs. RandomPlayer: {wins} wins, {losses} losses, {draws} draws")

    # (wins, losses, draws) = tournament(AlphaBetaPlayer(), GreedyPlayer(), games=100)
    # print(f"AlphaBetaPlayer vs. GreedyPlayer: {wins} wins, {losses} losses, {draws} draws")

    # (wins, losses, draws) = tournament(
    #     IteratedDeepeningAlphaBetaPlayer(decay=0.95),
    #     GreedyPlayer(),
    #     games=10)
    # print(f"IteratedDeepeningAlphaBetaPlayer vs. GreedyPlayer: {wins} wins, {losses} losses, {draws} draws")

    (wins, losses, draws) = tournament(
        MCTSPlayer(logger=None, max_nodes=1000),
        IteratedDeepeningAlphaBetaPlayer(decay=0.95),
        games=10)
    print(f"MCTSPlayer vs. IteratedDeepeningAlphaBetaPlayer: {wins} wins, {losses} losses, {draws} draws")

def tournament(player: Player, opponent: Player, games: int = 100) -> tuple[int, int, int]:
    """Play a tournament between two players,
    alternating the starting player in each game.
    Return the number of wins for each player and the number of draws."""

    players = {0: player, 1: opponent}
    wins = {0: 0, 1: 0}
    draws = 0
    for i in range(games):
        logging.info(f"Game {i+1}/{games}")
        p = i % 2
        o = (i + 1) % 2
        outcome = play(players[p], players[o])
        if outcome.winner is None:
            draws += 1
        elif outcome.winner:
            wins[p] += 1
        else:
            wins[o] += 1
    return wins[0], wins[1], draws

def play(player1: Player, player2: Player) -> chess.Outcome:
    """Play a game between two players."""
    logging.info("Playing {} vs. {}".format(player1, player2))
    start = AtomicBoard()
    board = start.copy()
    outcome: chess.Outcome | None  = None
    while outcome is None:
        board.push(player1.move(board))
        outcome = board.outcome()
        if outcome is not None:
            break
        board.push(player2.move(board))
        outcome = board.outcome()
    game = start.variation_san(board.move_stack)
    logging.info(f"Game: {game}")
    logging.info(f"Outcome: {outcome}")
    return outcome

if __name__ == "__main__":
    main()