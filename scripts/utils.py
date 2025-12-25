import subprocess

def run_solver(input_json, flags):
    cmd = ["build/cn", input_json] + flags
    subprocess.run(cmd, check=True)
