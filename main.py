import posixpath
import threading
from functools import partial
import argparse
import os
import signal
import orjson
import cluster
import sys
import socket
import subprocess
import json
import tempfile
from sweep import Sweep, extract_varying_keys, embellish_config
import time
import glob
import torch.profiler
import cProfile
import pstats
import io

MAX_JOBS = 100
jobs_queue = []
PID = os.getpid()

class JobSubmitter(threading.Thread):
    def __init__(self, total_jobs, sweep):
        super().__init__(daemon=True)
        self.stop_event = threading.Event()
        self.deer = f"{cluster.OUTPUT_HOME}/{PID}_run"
        os.makedirs(self.deer, exist_ok=True)
        print(f"{self.deer=}")
        self.total_jobs = total_jobs
        self.submitted_jobs = 0
        self.completed_jobs = 0
        self.sweep = sweep
        self.job_id_to_config = {}
        self.best_config = None
        self.best_objective = float('-inf')
        self.best_cost = float('inf')

    def run(self):
        self.last_job_queue_len = -1
        while (jobs_queue or self.completed_jobs < self.total_jobs) and not self.stop_event.is_set():
            current_jobs = int(subprocess.getoutput('squeue -h | wc -l'))
            if current_jobs < MAX_JOBS and jobs_queue:
                jobs_to_submit = min(MAX_JOBS - current_jobs, len(jobs_queue))
                for _ in range(jobs_to_submit):
                    job = jobs_queue.pop(0)
                    job_id = submit_job(*job)
                    self.job_id_to_config[job_id] = job[0]
                    self.submitted_jobs += 1
            
            self.check_completed_jobs()
            
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

    def check_completed_jobs(self):
        pattern = os.path.join(self.deer, "job_*_result.json")
        for filename in glob.glob(pattern):
            with open(filename, 'r') as f:
                result = json.load(f)
            job_id = result['job_id']
            objective = result['objective']
            cost = result['cost']
            config = self.job_id_to_config.pop(job_id)
            self.sweep.observe(config=config, value=objective, cost=cost)
            if objective > self.best_objective:
                self.best_config = config
                self.best_objective = objective
                self.best_cost = cost
            os.remove(filename)
            self.completed_jobs += 1

    def update_get_file(self):
        get_file = f"{self.deer}/get"
        with open(get_file, 'w') as f:
            progress = self.completed_jobs / self.total_jobs
            bar_length = 50
            filled_length = int(bar_length * progress)
            bar = '=' * filled_length + '-' * (bar_length - filled_length)
            f.write(f"\nProgress: [{bar}] {self.completed_jobs}/{self.total_jobs}")
            f.write(f"\nSubmitted jobs: {self.submitted_jobs}")
            if self.best_config:
                f.write(f"\nBest config: {json.dumps(extract_varying_keys(self.sweep.config, self.best_config))}")
                f.write(f"\nBest objective: {self.best_objective}")
                f.write(f"\nBest cost: {self.best_cost}")

def submit_job(config, use_tempfile, extra_args):
    job_name = f"run_{PID}"
    output_file = f"{job_name}_%j.txt"
    
    if use_tempfile:
        with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.json') as temp_config_file:
            json.dump(config, temp_config_file)
            config_path = temp_config_file.name
    else:
        config_path = json.dumps(config)

    cmd = f"python {posixpath.join(cluster.CODE_HOME, 'main.py')} --config '{config_path}' {extra_args}"
    job_id = cluster.run(cmd, filename=output_file, job_name=job_name)
    return job_id

def run_slurm(config, use_tempfile, extra_args):
    jobs_queue.append((config, use_tempfile, extra_args))

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

def instantiate_from_config(config, *args, recursive=True, curry=False, variables=None, **kwargs):
    if variables is None:
        variables = {}
    
    if isinstance(config, dict) and '_' in config:
        cls = variables.get(config['_']) or globals().get(config['_'])
        if cls is None:
            raise ValueError(f"Class {config['_']} not found in variables or globals")
        if recursive:
            params = {k: instantiate_from_config(v, recursive=recursive, curry=curry, variables=variables) for k, v in config.items() if k != '_'}
        else:
            params = {k: v for k, v in config.items() if k != '_'}
        if curry:
            return partial(cls, *args, **kwargs, **params)
        else:
            return cls(*args, **kwargs, **params)
    elif isinstance(config, dict) and recursive:
        return {k: instantiate_from_config(v, recursive=recursive, curry=curry, variables=variables) for k, v in config.items()}
    elif isinstance(config, list) and recursive:
        return [instantiate_from_config(item, recursive=recursive, curry=curry, variables=variables) for item in config]
    return config

def save_model(trainer, signal_name=None):
    model_filename = posixpath.join(cluster.OUTPUT_HOME, "model.pt")
    if model_filename:
        import torch
        best = dict()
        if trainer.best:
            best |= trainer.best
        if signal_name:
            best['interrupted_signal'] = signal_name
        if best:
            print(f"Saving model to {model_filename}")
            torch.save(best, model_filename)

interrupted_signal = None
trainer = config_ = None
def signal_handler(signum, frame):
    global interrupted_signal, config_, trainer
    interrupted_signal = signal.Signals(signum).name
    if config_ and trainer:
        save_model(trainer, interrupted_signal)
    print(f"Training interrupted by signal: {interrupted_signal}")
    exit(0)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def train(config):
    global trainer, config_
    
    import traceback
    import torch
    import meth
    from env import create_rssm_env, DMCEnv
    import nets
    import actor
    import critic
    from logger import SimpleLogger
    import trainer
    import trainer.objective

    # Dynamic imports
    for module in ['nets', 'actor', 'critic', 'trainer', 'trainer.objective']:
        module_obj = __import__(module, fromlist=['*'])
        globals().update({name: getattr(module_obj, name) for name in dir(module_obj) if not name.startswith('_')})
    from replay import ReplayBuffer
    import data
    import visuals
    from trainer import ClippedAdam

    config_ = config

    if config.get('seed', None) is not None:
        meth.seed_all(config['seed'])

    is_test = config.get('is_test', False)
    if is_test:
        config['project_name'] = None
        config['num_epochs'] = 2
        config['num_batches'] = 2
        config['batch_size'] = 2
        config['seq_len'] = 2
        config['num_episodes'] = 2
        config['eval_epochs'] = 1

    # Create logger
    logger = SimpleLogger(project_name=config.get('project_name'), config=config)

    # Create replay buffer
    replay = ReplayBuffer()
    
    # Filter and load data
    img_size = config['img_size']
    filtered_paths = data.filter_data(env=config['env_name'], collection_type='medium_replay', relation='main', px=img_size)
    if not filtered_paths:
        raise ValueError(f"No data found for environment: {config['env_name']}")
    
    loaded_data = data.load_deer(filtered_paths[0], num_episodes=config['num_episodes'])
    print(f"Loaded episodes: {loaded_data}")
    replay.add(loaded_data)

    # Create environment
    dmc_env = DMCEnv(config['env_name'], obs_type=config['obs_type'], img_size=img_size, action_repeat=config.get('action_repeat', 1))
    env = instantiate_from_config(config['model_env'], dmc_env, curry=True, variables=locals())()

    # Create actor
    actor = Cat(
        instantiate_from_config(config['actor'], env.markov_state_space, env.action_space, variables=locals()),
        keys=['hidden', 'state'],
        target_keys=['action', 'log_prob'],
    )

    # Create critic
    critic = instantiate_from_config(config['critic'], env.markov_state_space, env.action_space, variables=locals())

    # Create objectives
    objectives = {}
    for obj_name in ['model', 'actor', 'critic']:
        obj_config = config.get(f'{obj_name}_objective', {})
        if '_' in obj_config:
            objectives[obj_name] = instantiate_from_config(obj_config, env=env, actor=actor, critic=critic, variables=locals())
        else:
            scales = obj_config.get('scales', None)
            if obj_name == 'model':
                objectives[obj_name] = RSSMEnvObjective(actor=actor, critic=critic, env=env, scales=scales)
            elif obj_name == 'actor':
                objectives[obj_name] = SACActorObjective(actor=actor, critic=critic, env=env, scales=scales)
            elif obj_name == 'critic':
                objectives[obj_name] = SACCriticObjective(actor=actor, critic=critic, env=env, scales=scales)

    # Create optimizers
    optimizers = {
        'model': ClippedAdam(env.parameters(), **config['model_optimizer']),
        'actor': ClippedAdam(actor.parameters(), **config['actor_optimizer']),
        'critic': ClippedAdam(critic.parameters(), **config['critic_optimizer']),
    }

    # Create scheduler
    scheduler = instantiate_from_config(config['scheduler'], variables=locals())

    # Create trainer
    trainer = instantiate_from_config(
        config['trainer'],
        env=dmc_env,
        model=env,
        actor=actor,
        critic=critic,
        replay=replay,
        scheduler=scheduler,
        optimizers=optimizers,
        objectives=objectives,
        logger=logger,
        variables=locals()
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer.to(device)

    # Train
    def new_best_model_cb():
        if not is_test:
            save_model(trainer)
            static_frames = visuals.visualize_best_model_static(
                env=dmc_env,
                model=env,
                actor=actor,
                critic=critic,
                max_len=100,
            )
            if not cluster.get_is_cluster() and config.get('save_video', False):
                visuals.save_frames_as_video(
                    static_frames,
                    output_path=posixpath.join(cluster.OUTPUT_HOME, "video.mp4")
                )
            return dict(video=static_frames)
        return dict()

    start_time = time.time()
    
    if config.get('profile', False):
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                # torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            # on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/profiler'),
            record_shapes=True,
            profile_memory=False,
            with_stack=False
        ) as prof:
            best_eval_return = trainer.train(
                epochs=8,
                batches=4,
                bs=4,
                sl=4,
                eval_epochs=12345,
                new_best_model_cb=None,
                profiler=prof,
            )
        
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    else:
        best_eval_return = trainer.train(
            epochs=config['num_epochs'],
            batches=config['num_batches'],
            bs=config['batch_size'],
            sl=config['seq_len'],
            eval_epochs=config['eval_epochs'],
            new_best_model_cb=new_best_model_cb,
        )
    
    end_time = time.time()
    training_time = end_time - start_time

    logger.finish()

    # If running on cluster, save results to a file
    if cluster.get_is_cluster():
        result = {
            'job_id': os.environ.get('SLURM_JOB_ID'),
            'objective': best_eval_return,
            'cost': training_time
        }
        result_file = f"{cluster.OUTPUT_HOME}/{PID}_run/job_{result['job_id']}_result.json"
        with open(result_file, 'w') as f:
            json.dump(result, f)

    return best_eval_return

def parse_args():
    parser = argparse.ArgumentParser(description="Train model with given configuration.")
    parser.add_argument('--config', type=str, help="Path to the configuration file or serialized JSON.")
    parser.add_argument('-t', '--is_test', action='store_true', help="Set to run in test mode")
    parser.add_argument('--sweep', type=str, help="Path to sweep JSON file, serialized JSON string, or direct JSON containing sweep_id, project_name")
    parser.add_argument('-a', '--agent', action='store_true', help="Set to run agent with configs/sweep.json")
    parser.add_argument('-ut', '--use-tempfile', action='store_true', help="Use tempfile for config instead of direct JSON.")
    parser.add_argument('-e', '--extra_args', nargs=argparse.REMAINDER, help="Additional arguments to pass through.")
    parser.add_argument('-p', '--profile', action='store_true', help="Enable profiling")
    args = parser.parse_args()
    return args

def load_json(json_input):
    if os.path.exists(json_input) and os.path.isfile(json_input):
        with open(json_input, 'r') as f:
            return orjson.loads(f.read())
    else:
        try:
            return orjson.loads(json_input)
        except orjson.JSONDecodeError:
            raise ValueError("Invalid input: not a valid file path or JSON string")

if __name__ == "__main__":
    args = parse_args()
    
    config = None
    if args.config:
        config = load_json(args.config)

    if args.agent:
        cluster.run(f'{sys.executable} {__file__} --sweep configs/sweep.json')

    elif args.sweep:
        try:
            import wandb
            sweep = load_json(args.sweep)
            sweep_id = sweep['sweep_id']
            project_name = sweep['project_name']
            
            def _():
                wandb.init(project=project_name)
                print("Starting agent training loop...")
                train(wandb.config.as_dict() | dict(project_name=project_name, is_test=args.is_test, profile=args.profile))
            wandb.agent(sweep_id, function=_, project=project_name)

        except KeyError as e:
            raise ValueError(f"Missing key in sweep JSON: {e}")
            
    elif config:
        if 'method' in config and 'parameters' in config:

            config['parameters']['is_test'] = dict(value=args.is_test)
            config['parameters']['profile'] = dict(value=args.profile)

            project_name = config.pop("project_name", "uncategorized")

            import wandb
            sweep_id = wandb.sweep(sweep=config, project=project_name)

            sweep_config = dict(sweep_id=sweep_id, project_name=project_name)

            data = orjson.dumps(sweep_config).decode()
            with open('configs/sweep.json', 'w') as f:
                f.write(data)
            print(f"dumped {data} to configs/sweep.json")

        else:
        
            config['is_test'] = args.is_test
            config['profile'] = args.profile
            
            # Treat the config as a sweep config
            sweep = Sweep(config)
            if bool(sweep):
                sweep_configs = [sweep.suggest() for _ in range(int(1e4))]

                is_cluster = cluster.get_is_cluster()

                if is_cluster:
                    submitter = JobSubmitter(len(sweep_configs), sweep)
                    for config in sweep_configs:
                        run_slurm(config, args.use_tempfile, " ".join(args.extra_args) if args.extra_args else "")
                    submitter.start()
                    submitter.join()
                    submitter.cleanup()
                else:
                    for i, config in enumerate(sweep_configs, 1):
                        print(f"Suggested config: {json.dumps(embellish_config(extract_varying_keys(sweep.config, config)), indent=4)}")
                        print(f"Progress: {i}/{len(sweep_configs)}")
                        process = run_local(config, args.use_tempfile, " ".join(args.extra_args) if args.extra_args else "")
                        if process.returncode != 0:
                            print(f"Process failed with return code: {process.returncode}")
                            break
            else:
                config = sweep.suggest()  # Generate only one configuration
                train(config | dict(is_test=args.is_test, profile=args.profile))  # Immediately run using train
