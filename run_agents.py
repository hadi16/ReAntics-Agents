from my_agents import MY_AGENTS
import os
from zipfile import ZipFile


# If the Game module doesn't exist,
# must extract the ReAntics source code from the ZIP file.
try:
    import Game
except ImportError:
    with ZipFile('ReAntics.zip', 'r') as zipped_reantics_source:
        zipped_reantics_source.extractall(os.getcwd())

    # Move all of my agents to the AI/ folder.
    for agent in MY_AGENTS:
        os.rename(agent, 'AI/'+agent)

    # Now, try to import the Game module again.
    import Game

# Start the ReAntics game.
Game.Game()
