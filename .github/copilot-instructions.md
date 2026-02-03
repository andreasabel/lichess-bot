# Copilot Instructions for lichess-bot

## Overview

lichess-bot is a bridge between the Lichess Bot API and chess engines. It supports UCI, XBoard, and homemade Python engines, enabling bots to play on lichess.org.

## Build, Test, and Lint

### Install Dependencies
```bash
pip install -r requirements.txt
pip install -r test_bot/test-requirements.txt
```

### Run Tests
```bash
# Run all tests
pytest --log-cli-level=10

# Run a specific test
pytest test_bot/test_bot.py::test_homemade -v

# Run a specific test file
pytest test_bot/test_model.py -v
```

### Type Checking
```bash
mypy --strict .
```

### Run the Bot
```bash
python lichess-bot.py
```

## Architecture

### Core Components

**Main Entry Point**: `lichess-bot.py` → imports and calls `lib.lichess_bot.start_program()`

**Key Modules**:
- `lib/lichess_bot.py` - Main bot control loop, event stream handling, game coordination
- `lib/engine_wrapper.py` - Unified interface for UCI/XBoard/Homemade engines, opening books, tablebases, draw/resign logic
- `lib/lichess.py` - Lichess API client (REST + event streaming)
- `lib/model.py` - Data models for Game, Challenge, etc.
- `lib/config.py` - Configuration loading and validation from config.yml
- `lib/matchmaking.py` - Bot challenge creation logic

**Engine Types**:
1. **UCI/XBoard** - External engines via `chess.engine` (Stockfish, Leela, etc.)
2. **Homemade** - Python-based engines in `homemade.py` that inherit from `MinimalEngine`

### Game Flow

1. Bot starts → loads `config.yml` → upgrades account to BOT if needed
2. Main thread watches Lichess event stream (`watch_control_stream`)
3. On `gameStart` event → spawns multiprocessing worker to play game
4. Each game runs independently in `play_game()` with its own engine instance
5. Game loop: poll game state stream → engine.search() → send move → repeat
6. Engines use opening books, tablebases, and online move sources before searching
7. Draw/resign logic evaluated after each move based on score thresholds

### Homemade Engines

The `homemade.py` file demonstrates how to create custom Python engines. Key points:
- Inherit from `MinimalEngine` (defined in `lib/engine_wrapper.py`)
- Implement `search(board, *args)` returning `PlayResult(move, ponder)`
- Custom engine implementations in `engines/` provide reusable components:
  - `engines/player.py` - Base player classes (RandomPlayer, GreedyPlayer)
  - `engines/alphabeta.py` - Alpha-beta pruning with iterative deepening
  - `engines/mcts.py` - Monte Carlo Tree Search
  - `engines/alphazero.py` - Neural network-based evaluation

## Key Conventions

### Configuration
- Main config: `config.yml` (copy from `config.yml.default`)
- Token stored in `token` field or separate file referenced by path
- Engine protocol: `uci`, `xboard`, or `homemade`
- For homemade engines, set `engine.name` to the class name in `homemade.py`

### Multiprocessing Architecture
- Main process handles event stream and matchmaking
- Games run in separate processes (multiprocessing.Pool)
- Queues communicate between processes: control_queue, correspondence_queue, logging_queue, pgn_queue
- Type hints use custom types: `MULTIPROCESSING_LIST_TYPE`, `LICHESS_TYPE`, `POOL_TYPE`

### Engine Wrapper Pattern
- `create_engine()` returns context manager - always use in `with` block
- Engine wrapper handles:
  - Opening book moves (polyglot, online sources)
  - Tablebase queries (local syzygy/gaviota, online EGTB)
  - Time management and move overhead
  - Draw offers and resign logic based on evaluation scores
- Move priority: opening books → tablebases → engine search

### Type Safety
- Strict mypy typing enforced (`mypy --strict .`)
- Custom type definitions in `lib/lichess_types.py`
- Use TypedDict for complex dictionaries
- Use Union types for Lichess/test mocks

### Testing
- Mock Lichess API in `test_bot/lichess.py` for offline testing
- Tests download engines to `TEMP/` directory (cached in CI)
- Tests require `LICHESS_BOT_TEST_TOKEN` environment variable for some tests
- Use `disable_restart()` in tests to prevent automatic restarts on errors

### Signal Handling
- SIGINT (Ctrl+C) triggers graceful shutdown via `terminated` flag
- Second SIGINT forces immediate exit via `force_quit` flag
- Global state managed through module-level variables in `lib/lichess_bot.py`

### Versioning
- Version info in `lib/versioning.yml`
- Contains: bot version, minimum Python version, deprecation warnings
- Loaded at module import time into `__version__`

### Custom Game Logic
- `extra_game_handlers.py` provides hooks:
  - `game_specific_options(game)` - Return engine options per game
  - `is_supported_extra(challenge)` - Additional challenge acceptance logic
- These functions are called even if empty (no-op by default)

### Logging
- Uses Python's `logging` module with RichHandler for console output
- Logger instances: `logging.getLogger(__name__)` in each module
- Log levels controlled by config and command-line args
- Separate logging queue for multiprocess coordination
