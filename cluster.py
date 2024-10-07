import os
import socket
import subprocess
import argparse


CODE_HOME = os.path.dirname(os.path.abspath(__file__))
PROJ_HOME = os.path.dirname(CODE_HOME)
DATA_HOME = os.path.join(PROJ_HOME, 'data')
OUTPUT_HOME = os.path.join(PROJ_HOME, 'output')


def get_is_cluster():
    return 'uc2n' in socket.gethostname()

def run(cmd, is_cluster=None, filename='sweep.txt', job_name='sweep', time=1):
    times = [
        '00:00:30',
        '15:59:59',
        '23:59:59',
    ]
    if is_cluster is None:
        is_cluster = get_is_cluster()
    if is_cluster:
        cmd = [
            "sbatch",
            "--partition=gpu_4,gpu_8,gpu_4_a100,gpu_4_h100",
            f"--time={times[time]}",
            "--gres=gpu:1",
            "--mem-per-cpu=2000MB",
            "--cpus-per-task=1",
            "--job-name", job_name,
            "--output", f"{PROJ_HOME}/output/{filename}",
            "--error", f"{PROJ_HOME}/output/{filename}",
            "--wrap", f"""
            #!/bin/bash
            export PROJ_HOME={PROJ_HOME}
            export MUJOCO_GL="egl"
            echo "PROJ_HOME is set to $PROJ_HOME"
            source $PROJ_HOME/pyvenv/bin/activate
            export PYTHONPATH=$PYTHONPATH:{CODE_HOME}
            {cmd}
            """
        ]
        print("Submitting SLURM job with command:\n", '\n'.join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        job_id = result.stdout.strip().split()[-1]
        return job_id
    else:
        cmd = f"export PYTHONPATH=$PYTHONPATH:{CODE_HOME}; export MUJOCO_GL='egl'; {cmd}"
        return subprocess.run(cmd, shell=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a command on a cluster or locally.")
    parser.add_argument("cmd", help="The command to run")
    parser.add_argument("-f", "--filename", default="sweep.txt", help="Output filename")
    parser.add_argument("-j", "--job_name", default="sweep", help="Job name")
    parser.add_argument("-t", "--time", type=int, choices=[0, 1, 2], default=1, help="Time limit index")
    
    args = parser.parse_args()
    
    run(args.cmd, is_cluster=None, filename=args.filename, job_name=args.job_name, time=args.time)
