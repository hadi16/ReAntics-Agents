from all_agents import all_agents
import os
from zipfile import ZipFile


try:
    import Game
    Game.Game()
except ImportError:
    with ZipFile('ReAntics.zip', 'r') as zip_reference:
        zip_reference.extractall(os.getcwd())

    for agent in all_agents:
        os.rename(agent, 'AI/'+agent)

    import Game
    Game.Game()
