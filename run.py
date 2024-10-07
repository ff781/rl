import cluster
import os
import socket
import subprocess
import argparse
import json
import threading
import itertools
import tempfile
import signal
import sys
import posixpath

PROJ_HOME = '/pfs/data5/home/kit/stud/ujtsk/proj/bisch0'
if not os.path.exists(PROJ_HOME):
    PROJ_HOME = os.environ.get('PROJ_HOME', os.path.expanduser('~'))
MAX_JOBS = 100
jobs_queue = []
PID = os.getpid()


class JobSubmitter(threading.Thread):
    def __init__(self, total_jobs):
        super().__init__(daemon=True)
        self.stop_event = threading.Event()
        self.deer = f"{PROJ_HOME}/output/run_{PID}"
        os.makedirs(self.deer, exist_ok=True)
        self.total_jobs = total_jobs
        self.submitted_jobs = 0

    def run(self):
        self.last_job_queue_len = -1
        while jobs_queue and not self.stop_event.is_set():
            current_jobs = int(subprocess.getoutput('squeue -h | wc -l'))
            if current_jobs < MAX_JOBS:
                jobs_to_submit = min(MAX_JOBS - current_jobs, len(jobs_queue))
                for _ in range(jobs_to_submit):
                    job = jobs_queue.pop(0)
                    submit_job(*job)
                    self.submitted_jobs += 1
            if len(jobs_queue) != self.last_job_queue_len:
                self.last_job_queue_len = len(jobs_queue)
                self.update_get_file()
            self.stop_event.wait(8)

    def stop(self):
        self.stop_event.set()

    def cleanup(self):
        if self.deer is not None:
            import shutil
            shutil.rmtree(self.deer)

    def update_get_file(self):
        get_file = f"{PROJ_HOME}/output/run_{PID}/get"
        with open(get_file, 'w') as f:
            # for job in jobs_queue:
            #     f.write(json.dumps(job[0], indent=4))  # Write the imputed_config
            #     f.write("\n\n")  # Add blank lines between jobs
            
            # Add progress bar and count
            progress = self.submitted_jobs / self.total_jobs
            bar_length = 50
            filled_length = int(bar_length * progress)
            bar = '=' * filled_length + '-' * (bar_length - filled_length)
            f.write(f"\nProgress: [{bar}] {self.submitted_jobs}/{self.total_jobs}")


import copy
def unravel_sweep(config):
    if isinstance(config, dict):
        extra_keys = [{}]
        if '_keys' in config:
            extra_keys = unravel_sweep(config.pop('_keys'))
        a = list(map(unravel_sweep, config.values()))
        list_items = [(k, v) for k, v in zip(config.keys(), a) if isinstance(v, list)]
        b, c = zip(*list_items) if list_items else ([], [])
        a = {k: (None if isinstance(v, list) else v) for k, v in zip(config.keys(), a)}
        products = list(itertools.product(*c)) if c else [()]
        result = []
        for extra_key in extra_keys:
            for product in products:
                new_a = copy.deepcopy(a)
                new_a.update(extra_key)
                for k, value in zip(b, product):
                    new_a[k] = value
                result.append(new_a)
        return result
    elif isinstance(config, tuple):
        a = list(map(unravel_sweep, config))
        list_items = [(i, v) for i, v in enumerate(a) if isinstance(v, list)]
        if not list_items:
            return [config]
        b, c = zip(*list_items)
        a = list(map(lambda x: None if isinstance(x, list) else x, a))
        products = list(itertools.product(*c))
        result = []
        for product in products:
            new_a = copy.deepcopy(a)
            for i, value in zip(b, product):
                new_a[i] = value
            result.append(tuple(new_a))
        return result
    elif isinstance(config, list):
        return sum(map(lambda a: a if isinstance(a,list) else [a], map(unravel_sweep, config)), start=[])
    return config

def impute(framework, config):
    def _impute(framework, config):
        if isinstance(framework, dict) and isinstance(config, dict):
            for k, v in framework.items():
                if k in config:
                    framework[k] = _impute(v, config[k])
        return config
    framework = copy.deepcopy(framework)
    _impute(framework, config)
    return framework

def red(config):
    if isinstance(config, dict):
        return dict(filter(lambda a: a[1] is not None, map(lambda a: (a[0], red(a[1])), config.items())))
    elif isinstance(config, tuple):
        return tuple(red(a) for a in config)
    elif isinstance(config, list):
        value = None
        a = dict()
        for b in config:
            b = red(b)
            if isinstance(b, dict):
                a |= b
            elif value is None:
                value = b
        return a if a else value
    return config

def sweeps_only(config, ok=0):
    if not contains_sweep(config) and not ok:
        return None
    if isinstance(config, dict):
        return dict(filter(lambda a: a[1] is not None, map(lambda a: (a[0], sweeps_only(a[1])), config.items())))
    elif isinstance(config, tuple):
        return tuple(filter(lambda a: a is not None, map(sweeps_only, config)))
    elif isinstance(config, list):
        return config
    return config

def contains_sweep(config):
    if isinstance(config, dict):
        return '_keys' in config or any(contains_sweep(v) for v in config.values())
    elif isinstance(config, tuple):
        return any(contains_sweep(a) for a in config)
    elif isinstance(config, list):
        return True
    return False

def submit_job(imputed_config, config, use_tempfile, extra_args):
    job_name = f"run_{PID}"
    output_file = f"{job_name}_%j.txt"
    
    if use_tempfile:
        with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.json') as temp_config_file:
            json.dump(config, temp_config_file)
            config_path = temp_config_file.name
    else:
        config_path = json.dumps(config)

    cmd = f"python {posixpath.join(cluster.CODE_HOME, 'main.py')} --config '{config_path}' {extra_args}"
    cluster.run(cmd, filename=output_file, job_name=job_name)

def run_slurm(imputed_config, config, use_tempfile, extra_args):
    jobs_queue.append((imputed_config, config, use_tempfile, extra_args))

def run_local(config, use_tempfile, extra_args):
    if use_tempfile:
        with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.json') as temp_config_file:
            json.dump(config, temp_config_file)
            config_path = temp_config_file.name
    else:
        config_path = json.dumps(config)
    
    cmd = f"export PYTHONPATH=$PYTHONPATH:{cluster.CODE_HOME} && python {posixpath.join(cluster.CODE_HOME, 'main.py')} --config '{config_path}' {extra_args}"

    print(f"Running command: {cmd}")

    process = subprocess.run(cmd, shell=True)
    print(f"Process completed with return code: {process.returncode}")
    return process

def signal_handler(signum, frame):
    print("\nReceived Ctrl+C. Terminating all processes...")
    for process in running_processes:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    sys.exit(0)

def main():
    global submitter
    global args_
    global running_processes

    signal.signal(signal.SIGINT, signal_handler)
    running_processes = []

    parser = argparse.ArgumentParser(description="Select configuration and pass additional arguments.")
    parser.add_argument("config", type=str, help="Path to the configuration file or JSON string.")
    parser.add_argument("-ut", "--use-tempfile", action="store_true", help="Use tempfile for config instead of direct JSON.")
    parser.add_argument("-t", "--test", nargs="?", const=1, type=int, help="Run only the first n sweeps. If no number is provided, runs only the first sweep.")
    parser.add_argument("-e", "--extra_args", nargs=argparse.REMAINDER, help="Additional arguments to pass through.")
    args_ = parser.parse_args()

    if os.path.exists(args_.config) and os.path.isfile(args_.config):
        with open(args_.config, 'r') as f:
            base_config = json.load(f)
    else:
        try:
            base_config = json.loads(args_.config)
        except json.JSONDecodeError:
            raise ValueError("Invalid config: not a valid file path or JSON string")

    extra_args = " ".join(args_.extra_args) if args_.extra_args else ""

    sweep_configs = unravel_sweep(base_config)
    
    if args_.test is not None:
        sweep_configs = sweep_configs[:args_.test]
    
    only_sweeps = sweeps_only(base_config)
    reduced_config = red(only_sweeps)

    is_cluster = 'uc2n' in socket.gethostname()

    if is_cluster:
        submitter = JobSubmitter(len(sweep_configs))
        print(f"{submitter.deer=}")
        for config in sweep_configs:
            imputed_config = impute(reduced_config, config)
            run_slurm(imputed_config, config, args_.use_tempfile, extra_args)
        submitter.start()
        submitter.join()
        submitter.cleanup()
    else:
        for i, config in enumerate(sweep_configs, 1):
            imputed_config = impute(reduced_config, config)
            print(f"Imputed config: {json.dumps(imputed_config, indent=4)}")
            print(f"Progress: {i}/{len(sweep_configs)}")
            process = run_local(config, args_.use_tempfile, extra_args)
            if process.returncode != 0:
                print(f"Process failed with return code: {process.returncode}")
                break

if __name__ == "__main__":
    main()
