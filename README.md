# FPL Prediction Model

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![Pydantic v2](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/pydantic/pydantic/main/docs/badge/v2.json)](https://docs.pydantic.dev/latest/contributing/#badges)
![GitHub License](https://img.shields.io/github/license/Ch110pB/fantasypl)
![GitHub top language](https://img.shields.io/github/languages/top/Ch110pB/fantasypl)
![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2FCh110pB%2Ffantasypl%2Frefs%2Fheads%2Fmain%2Fpyproject.toml)

This repository is a collection of code for predicting an optimal squad for Fantasy Premier League.

[Fantasy Premier League](https://fantasy.premierleague.com/) (FPL, in short) is a game where one selects a total of 15
players (11 in lineup and 4 in bench) with certain restrictions for the selection. Each player receives points based on
their performance in the respective gameweeks of their teams in the Premier League. The objective of this predictive 
model is to predict the squad that will gain the most number of points.

## Parts of the Code

- Data Fetching: 
  -  The official FPL API is the first source of data which lists all the teams, all the players and a lot 
  of relevant details about each of them.
  - Next is [FBRef](https://fbref.com/en/), the primary source of information for this project related to all the stats
  for the teams and players, including but not limited to number of shots on target, number of key passes, number of 
  yellow cards and non-penalty expected goals.


- Data Processing and Model Training
  - With the data we have, the next step is to process the data to extract the required features for the Machine Learning
  model.
  - After that, the data goes through some pre-processing and train-test split, and then it is fed to an AutoML training
  to find the best model for the respective prediction.
  - The project currently is structured around a 2-Layer prediction, with the first layer predicting at the team level,
  and the next one predicting at the player level. The player level predictions are then normalized on the basis of the
  team level predictions.


- Optimization
  - Now with all the predicted points, we use Linear Optimization to find the best squad possible for FPL.
  - The final output is a nice Discord message with the squad image.
