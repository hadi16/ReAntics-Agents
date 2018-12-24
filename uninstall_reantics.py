from my_agents import MY_AGENTS
import os
import shutil


# All the files associated with ReAntics.
REANTICS_FILES = (
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
    'my-settings.json'  # Created when settings in ReAntics have been modified from defaults.
)

# All of the ReAntics folders.
REANTICS_FOLDERS = (
    'AI/',
    'manual/',
    'Textures/'
)

# First, move all of my agents out of the AI/ folder,
# so that they aren't deleted by the rmtree command.
for agent in MY_AGENTS:
    if os.path.exists('AI/'+agent):
        os.rename('AI/'+agent, agent)

# Delete all the ReAntics files.
for game_file in REANTICS_FILES:
    if os.path.exists(game_file):
        os.remove(game_file)

# Delete all the ReAntics folders.
for folder in REANTICS_FOLDERS:
    shutil.rmtree(folder, ignore_errors=True)
