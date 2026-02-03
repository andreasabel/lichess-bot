From the root: python -m engines.tournament

- GreedyPlayer vs. RandomPlayer: 92 wins, 8 losses, 0 draws
- AlphaBetaPlayer vs. GreedyPlayer: 96 wins, 4 losses, 0 draws
- IteratedDeepeningAlphaBetaPlayer vs. GreedyPlayer: 10 wins, 0 losses, 0 draws

MCTS without win/loss consideration in best move selection
- MCTSPlayer vs. IteratedDeepeningAlphaBetaPlayer: 4 wins, 6 losses, 0 draws
- MCTSPlayer vs. IteratedDeepeningAlphaBetaPlayer: 4 wins, 5 losses, 1 draws

MCTSPlayer vs. IteratedDeepeningAlphaBetaPlayer: 6 wins, 4 losses, 0 draws