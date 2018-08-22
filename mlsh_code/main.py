import argparse
import tensorflow as tf
parser = argparse.ArgumentParser()
'''basic save and log'''
parser.add_argument('--save-dir', default='../results/',
                    help='directory to save agent logs')
parser.add_argument('--exp', type=str,
                    help='exp')

parser.add_argument('--task', type=str)
parser.add_argument('--num_subs', type=int)
parser.add_argument('--macro_duration', type=int)
parser.add_argument('--num_rollouts', type=int)
parser.add_argument('--warmup_time', type=int)
parser.add_argument('--train_time', type=int)
parser.add_argument('--force_subpolicy', type=int)
parser.add_argument('--replay', type=str)
parser.add_argument('-s', action='store_true')
parser.add_argument('--continue_iter', type=str)

parser.add_argument('--log-interval', type=int, default=1,
                    help='log interval, one log per n updates')
parser.add_argument('--save-interval', type=int, default=100,
                    help='save interval, one save per n updates')
parser.add_argument('--vis-interval', type=int, default=1,
                    help='vis interval, one log per n updates')

'''environment details'''
parser.add_argument('--obs-type', type=str, default='image',
                    help='observation type: image or ram' )
parser.add_argument('--env-name', default='PongNoFrameskip-v4',
                    help='environment to train on')
parser.add_argument('--reward-level', type=int, default=2,
                    help='level of reward in games like OverCooked')
parser.add_argument('--use-fake-reward-bounty', action='store_true',
                    help='if use fake reward bounty')
parser.add_argument('--reset-leg', action='store_true',
                    help='if reset four legs after four steps')
parser.add_argument('--add-goal-color', action='store_true',
                    help='if add area color when get the goal')

'''for log behavior'''
parser.add_argument('--log-behavior-interval', type=int, default=10,
                    help='log behavior every x minutes')
parser.add_argument('--act-deterministically', action='store_true',
                    help='if act deterministically when interactiong')

parser.add_argument('--aux', type=str, default='',
                    help='some aux information you may want to record along with this run')

parser.add_argument('--test-reward-bounty', action='store_true',
                    help='to test what reward bounty will each macro-action produce')
parser.add_argument('--test-action', action='store_true',
                    help='specify actions at every level')
parser.add_argument('--test-action-vis', action='store_true',
                    help='see actions at every level')
parser.add_argument('--run-overcooked', action='store_true',
                    help='run overcooked to debug the game')
parser.add_argument('--see-leg-fre', action='store_true',
                    help='see the frequency of each leg through tensorboard')
parser.add_argument('--render', action='store_true',
                    help='render environment in a window')

args = parser.parse_args()

# python main.py --task MovementBandits-v0 --num_subs 2 --macro_duration 10 --num_rollouts 1000 --warmup_time 60 --train_time 1 --replay True test

from mpi4py import MPI
from rl_algs.common import set_global_seeds, tf_util as U
import os.path as osp
import gym, logging
import numpy as np
from collections import deque
from gym import spaces
import misc_util
import sys
import shutil
import subprocess
import master

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

replay = str2bool(args.replay)
args.replay = str2bool(args.replay)

'''basic save path'''
args.save_dir = osp.join(args.save_dir, args.exp)
args.save_dir = osp.join(args.save_dir, 'mlsh')

'''environment details'''
args.save_dir = osp.join(args.save_dir, 'o_t-{}'.format(args.obs_type))
args.save_dir = osp.join(args.save_dir, 'e_n-{}'.format(args.env_name))
if args.env_name in ['OverCooked']:
    args.save_dir = osp.join(args.save_dir, 'r_l-{}'.format(args.reward_level))
    args.save_dir = osp.join(args.save_dir, 'u_f_r_b-{}'.format(args.use_fake_reward_bounty))
    args.save_dir = osp.join(args.save_dir, 'r_lg-{}'.format(args.reset_leg))
    args.save_dir = osp.join(args.save_dir, 'a_g_c-{}'.format(args.add_goal_color))

RELPATH = osp.join(args.save_dir)
LOGDIR = osp.join('/root/results' if sys.platform.startswith('linux') else '/tmp', RELPATH)

def callback(it):
    if MPI.COMM_WORLD.Get_rank()==0:
        if it % 5 == 0 and it > 3 and not replay:
            fname = osp.join("savedir/", 'checkpoints', '%.5i'%it)
            U.save_state(fname)
    if it == 0 and args.continue_iter is not None:
        fname = osp.join("savedir/"+args.savename+"/checkpoints/", str(args.continue_iter))
        U.load_state(fname)
        pass

def train():
    num_timesteps=1e9
    seed = 1401
    rank = MPI.COMM_WORLD.Get_rank()
    sess = U.single_threaded_session()
    sess.__enter__()
    workerseed = seed + 1000 * MPI.COMM_WORLD.Get_rank()
    rank = MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed)

    # if rank != 0:
    #     logger.set_level(logger.DISABLED)
    # logger.log("rank %i" % MPI.COMM_WORLD.Get_rank())

    world_group = MPI.COMM_WORLD.Get_group()
    mygroup = rank % 10
    theta_group = world_group.Incl([x for x in range(MPI.COMM_WORLD.size) if (x % 10 == mygroup)])
    comm = MPI.COMM_WORLD.Create(theta_group)
    comm.Barrier()
    # comm = MPI.COMM_WORLD

    master.start(callback, args=args, workerseed=workerseed, rank=rank, comm=comm)

def main():
    if MPI.COMM_WORLD.Get_rank() == 0 and osp.exists(LOGDIR):
        shutil.rmtree(LOGDIR)
    MPI.COMM_WORLD.Barrier()
    # with logger.session(dir=LOGDIR):
    train()

if __name__ == '__main__':
    main()
