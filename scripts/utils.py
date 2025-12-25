import subprocess
import os 

def run_solver(input_json, flags, platform):
    env = os.environ.copy()
    env["OCL_PLATFORM"] = str(platform)

    cmd = ["build/cn", input_json] + flags
    subprocess.run(cmd, env=env, check=True)
