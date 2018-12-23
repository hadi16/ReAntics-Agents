from all_agents import all_agents
import os
import shutil


reantics_files = (
    'AIPlayerUtils.py',
    'Ant.py',
    'Building.py',
    'Constants.py',
    'Construction.py',
    'GUIHandler.py',
    'Game.py',
    'GamePane.py',
    'GameState.py',
    'HumanPlayer.py',
    'InfoScraper.py',
    'Inventory.py',
    'Location.py',
    'Move.py',
    'Player.py',
    'RedoneWidgets.py',
    'SettingsPane.py',
    'StatsPane.py',
    'my-settings.json'
)

reantics_folders = (
    'AI/',
    'manual/',
    'Textures/'
)

for agent in all_agents:
    os.rename('AI/'+agent, agent)

for file in reantics_files:
    if os.path.exists(file):
        os.remove(file)

for folder in reantics_folders:
    shutil.rmtree(folder, ignore_errors=True)
