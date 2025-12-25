import utils

flags = ["--profile", "--delta-save=9"]

inputs = [
    "experiments_config/dot_config.json"
]

for inp in inputs:
    utils.run_solver(inp, flags);
