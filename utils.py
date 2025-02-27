import argparse
from collections import namedtuple
from datetime import datetime

from matplotlib import pyplot as plt
import wandb

from envs.ant import Ant
from envs.debug_env import Debug
from envs.half_cheetah import Halfcheetah
from envs.reacher import Reacher
from envs.pusher import Pusher, PusherReacher
from envs.ant_ball import AntBall
from envs.ant_maze import AntMaze
from envs.humanoid import Humanoid
from envs.ant_push import AntPush

Config = namedtuple(
    "Config",
    "debug discount unroll_length episode_length repr_dim random_goals disable_entropy_actor use_traj_idx_wrapper",
)


def create_parser():
    parser = argparse.ArgumentParser(description="Training script arguments")
    parser.add_argument("--exp_name", type=str, default="test", help="Name of the wandb experiment")
    parser.add_argument("--group_name", type=str, default="test", help="Name of the wandb group of experiment")
    parser.add_argument("--project_name", type=str, default="crl", help="Name of the wandb project of experiment")
    parser.add_argument("--num_timesteps", type=int, default=1000000, help="Number of training timesteps")
    parser.add_argument("--max_replay_size", type=int, default=10000, help="Maximum size of replay buffer")
    parser.add_argument("--min_replay_size", type=int, default=8192, help="Minimum size of replay buffer")
    parser.add_argument("--num_evals", type=int, default=50, help="Number of evaluations")
    parser.add_argument("--episode_length", type=int, default=53, help="Maximum length of each episode")
    parser.add_argument("--action_repeat", type=int, default=2, help="Number of times to repeat each action")
    parser.add_argument("--discounting", type=float, default=0.997, help="Discounting factor for rewards")
    parser.add_argument("--num_envs", type=int, default=128, help="Number of environments")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for training")
    parser.add_argument("--seed", type=int, default=0, help="Seed for reproducibility")
    parser.add_argument("--unroll_length", type=int, default=50, help="Length of the env unroll")
    parser.add_argument("--multiplier_num_sgd_steps", type=int, default=1, help="Multiplier of total number of gradient steps resulting from other args.",)
    parser.add_argument("--env_name", type=str, default="reacher", help="Name of the environment to train on")
    parser.add_argument("--normalize_observations", default=False, action="store_true", help="Whether to normalize observations")
    parser.add_argument("--log_wandb", default=False, action="store_true", help="Whether to log to wandb")
    parser.add_argument('--policy_lr', type=float, default=6e-4)
    parser.add_argument('--alpha_lr', type=float, default=3e-4)
    parser.add_argument('--critic_lr', type=float, default=3e-4)
    parser.add_argument('--contrastive_loss_fn', type=str, default='symmetric_infonce')
    parser.add_argument('--energy_fn', type=str, default='l2')
    parser.add_argument('--backend', type=str, default=None)
    parser.add_argument('--no_resubs', default=False, action='store_true', help="Not use resubstitution (diagonal) for logsumexp in contrastive cross entropy")
    parser.add_argument('--use_ln', default=False, action='store_true', help="Whether to use layer normalization for preactivations in hidden layers")
    parser.add_argument('--use_c_target', default=False, action='store_true', help="Use learnable c_target param in contrastive loss")
    parser.add_argument('--logsumexp_penalty', type=float, default=0.0)
    parser.add_argument('--l2_penalty', type=float, default=0.0)
    parser.add_argument('--exploration_coef', type=float, default=0.0)
    parser.add_argument('--random_goals', type=float, default=0.0, help="Propotion of random goals to use in the actor loss")
    parser.add_argument('--train_horizon', type=int, default=50, help="Number of steps to train on")
    parser.add_argument('--disable_entropy_actor', default=False, action="store_true", help="Whether to disable entropy in actor")
    parser.add_argument('--use_traj_idx_wrapper', default=False, action="store_true", help="Whether to use debug wrapper with info about envs, seeds and trajectories")
    parser.add_argument('--eval_env', type=str, default=None, help="Whether to use separate environment for evaluation")
    parser.add_argument("--h_dim", type=int, default=256, help="Width of hidden layers")
    parser.add_argument("--n_hidden", type=int, default=2, help="Number of hidden layers")
    parser.add_argument('--repr_dim', type=int, default=64, help="Dimension of the representation")
    return parser


def create_env(args: argparse.Namespace, **kwargs) -> object:
    env_name = args.env_name
    if env_name == "reacher":
        env = Reacher(backend=args.backend or "generalized")
    elif env_name == "ant":
        if kwargs.get("mode", "train") == "eval":
            dist = 50
        else:
            dist = 15
        env = Ant(backend=args.backend or "spring", goal_dist=dist)
    elif env_name == "ant_ball":
        env = AntBall(backend=args.backend or "spring")
    elif env_name == "ant_push":
        # This is stable only in mjx backend
        assert args.backend == "mjx"
        env = AntPush(backend=args.backend)
    elif "maze" in env_name:
        if kwargs.get("mode", "train") == "eval":
            dist = 200
        else:
            dist = 20
        env = AntMaze(backend=args.backend or "spring", maze_layout_name=env_name[4:], train_horizon=dist, **kwargs)
    elif env_name == "cheetah":
        env = Halfcheetah()
    elif env_name == "debug":
        env = Debug(backend=args.backend or "spring")
    elif env_name == "pusher_easy":
        env=Pusher(backend=args.backend or "generalized", kind="easy")
    elif env_name == "pusher_hard":
        env=Pusher(backend=args.backend or "generalized", kind="hard")
    elif env_name == "pusher_reacher":
        env=PusherReacher(backend=args.backend or "generalized")
    elif env_name == "humanoid":
        if kwargs.get("mode", "train") == "eval":
            dist = 50
        else:
            dist = 10
        env=Humanoid(backend=args.backend or "generalized", goal_dist=dist)
    else:
        raise ValueError(f"Unknown environment: {env_name}")
    return env


def create_eval_env(args: argparse.Namespace) -> object:
    # if not args.eval_env:
    #     return None
    
    eval_arg = argparse.Namespace(**vars(args))
    eval_arg.env_name = args.env_name
    return create_env(eval_arg, mode="eval")

def get_env_config(args: argparse.Namespace):
    if args.env_name == "debug":
        config = Config(
            debug=True,
            discount=args.discounting,
            unroll_length=args.unroll_length,
            episode_length=args.episode_length,
            repr_dim=args.repr_dim,
            random_goals=args.random_goals,
            disable_entropy_actor=args.disable_entropy_actor,
            use_traj_idx_wrapper=args.use_traj_idx_wrapper
        )
    elif args.env_name == "reacher":
        config = Config(
            debug=False,
            discount=args.discounting,
            unroll_length=args.unroll_length,
            episode_length=args.episode_length,
            repr_dim=args.repr_dim,
            random_goals=args.random_goals,
            disable_entropy_actor=args.disable_entropy_actor,
            use_traj_idx_wrapper=args.use_traj_idx_wrapper
        )
    elif args.env_name == "cheetah":
        config = Config(
            debug=False,
            discount=args.discounting,
            unroll_length=args.unroll_length,
            episode_length=args.episode_length,
            repr_dim=args.repr_dim,
            random_goals=args.random_goals,
            disable_entropy_actor=args.disable_entropy_actor,
            use_traj_idx_wrapper=args.use_traj_idx_wrapper
        )
    elif args.env_name == "pusher_easy" or args.env_name == "pusher_hard":
        config = Config(
            debug=False,
            discount=args.discounting,
            unroll_length=args.unroll_length,
            episode_length=args.episode_length,
            repr_dim=args.repr_dim,
            random_goals=args.random_goals,
            disable_entropy_actor=args.disable_entropy_actor,
            use_traj_idx_wrapper=args.use_traj_idx_wrapper
        )
    elif args.env_name == "pusher_reacher":
        config = Config(
            debug=False,
            discount=args.discounting,
            unroll_length=args.unroll_length,
            episode_length=args.episode_length,
            repr_dim=args.repr_dim,
            random_goals=args.random_goals,
            disable_entropy_actor=args.disable_entropy_actor,
            use_traj_idx_wrapper=args.use_traj_idx_wrapper
        )
    elif args.env_name == "ant" or 'maze' in args.env_name:
        config = Config(
            debug=False,
            discount=args.discounting,
            unroll_length=args.unroll_length,
            episode_length=args.episode_length,
            repr_dim=args.repr_dim,
            random_goals=args.random_goals,
            disable_entropy_actor=args.disable_entropy_actor,
            use_traj_idx_wrapper=args.use_traj_idx_wrapper
        )
    elif args.env_name == "ant_push":
        config = Config(
            debug=False,
            discount=args.discounting,
            unroll_length=args.unroll_length,
            episode_length=args.episode_length,
            repr_dim=args.repr_dim,
            random_goals=args.random_goals,
            disable_entropy_actor=args.disable_entropy_actor,
            use_traj_idx_wrapper=args.use_traj_idx_wrapper
        )
    elif args.env_name == "ant_ball":
        config = Config(
            debug=False,
            discount=args.discounting,
            unroll_length=args.unroll_length,
            episode_length=args.episode_length,
            repr_dim=args.repr_dim,
            random_goals=args.random_goals,
            disable_entropy_actor=args.disable_entropy_actor,
            use_traj_idx_wrapper=args.use_traj_idx_wrapper
        )
    elif args.env_name == "humanoid":
        config = Config(
            debug=False,
            discount=args.discounting,
            unroll_length=args.unroll_length,
            episode_length=args.episode_length,
            repr_dim=args.repr_dim,
            random_goals=args.random_goals,
            disable_entropy_actor=args.disable_entropy_actor,
            use_traj_idx_wrapper=args.use_traj_idx_wrapper
        )
    else:
        raise ValueError(f"Unknown environment: {args.env_name}")
    return config


class MetricsRecorder:
    def __init__(self, num_timesteps):
        self.x_data = []
        self.y_data = {}
        self.y_data_err = {}
        self.times = [datetime.now()]

        self.max_x, self.min_x = num_timesteps * 1.1, 0

    def record(self, num_steps, metrics):
        self.times.append(datetime.now())
        self.x_data.append(num_steps)

        for key, value in metrics.items():
            if key not in self.y_data:
                self.y_data[key] = []
                self.y_data_err[key] = []

            self.y_data[key].append(value)
            self.y_data_err[key].append(metrics.get(f"{key}_std", 0))

    def log_wandb(self):
        data_to_log = {}
        for key, value in self.y_data.items():
            data_to_log[key] = value[-1]
        data_to_log["step"] = self.x_data[-1]
        wandb.log(data_to_log, step=self.x_data[-1])

    def plot_progress(self):
        num_plots = len(self.y_data)
        num_rows = (num_plots + 1) // 2  # Calculate number of rows needed for 2 columns

        fig, axs = plt.subplots(num_rows, 2, figsize=(15, 5 * num_rows))

        for idx, (key, y_values) in enumerate(self.y_data.items()):
            row = idx // 2
            col = idx % 2

            axs[row, col].set_xlim(self.min_x, self.max_x)
            axs[row, col].set_xlabel("# environment steps")
            axs[row, col].set_ylabel(key)
            axs[row, col].errorbar(self.x_data, y_values, yerr=self.y_data_err[key])
            axs[row, col].set_title(f"{key}: {y_values[-1]:.3f}")

        # Hide any empty subplots
        for idx in range(num_plots, num_rows * 2):
            row = idx // 2
            col = idx % 2
            axs[row, col].axis("off")
        plt.tight_layout()
        plt.show()

    def print_progress(self):
        for idx, (key, y_values) in enumerate(self.y_data.items()):
            try:
                print(f"step: {self.x_data[-1]}, {key}: {y_values[-1]:.3f} +/- {self.y_data_err[key][-1]:.3f}")
            except:
                pass

    def print_times(self):
        print(f"time to jit: {self.times[1] - self.times[0]}")
        print(f"time to train: {self.times[-1] - self.times[1]}")
