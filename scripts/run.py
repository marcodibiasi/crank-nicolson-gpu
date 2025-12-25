import utils

flags = ["--profile", "--delta-save=9"]

inputs = (
    ("experiments_config/dot_config.json", "1"),
)

for name, plat in inputs:
    utils.run_solver(name, flags, plat);
