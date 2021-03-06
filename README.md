# ReAntics Agents

Artificial Intelligence agents for the game ReAntics (https://github.com/amnuxoll/ReAntics)

DEPENDENCY
- Python 3.6 or later (this version required for type hints).

HOW TO RUN THE AGENTS
- Run the Python script run_agents.py
    * On Windows: python run_agents.py
    * On macOS: python3 run_agents.py

HOW TO REMOVE REANTICS FILES AFTER RUNNING AGENTS
- Run the Python script remove_reantics_files.py
    * On Windows: python remove_reantics_files.py
    * On macOS: python3 remove_reantics_files.py

FILES & PURPOSE
- ReAntics.zip
    * Contains the source code for the ReAntics game.

- 1-Heuristic.py
    * A heuristic agent.
- 2-InformedSearch.py
    * An informed search agent.
- 3-MinimaxAlphaBetaPruning.py
    * A minimax and alpha-beta pruning agent.
- 4-GeneticAlgorithms.py
    * A genetic algorithms agent (note: must run for multiple iterations to train).
- 5-NeuralNetworks.py
    * A neural networks agent (note: the agent was trained and hardcoded weights were added).
- 6-TDLearning.py
    * A temporal difference learning agent.
- 6-state_utilities.json
    * The states with their associated utilities for the TD Learning agent to use.
