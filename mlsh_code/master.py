import gym
import test_envs
import tensorflow as tf
import rollouts
from policy_network import Policy
from subpolicy_network import SubPolicy
from observation_network import Features
from learner import Learner
import rl_algs.common.tf_util as U
import numpy as np
# from tinkerbell import logger
import pickle
import time

def start(callback, args, workerseed, rank, comm, logdir):
    if args.task in ['OverCooked']:
        import overcooked
        env = overcooked.OverCooked(
            args = args,
        )
    else:
        env = gym.make(args.task)

    if rank==0:
        summary_writer = tf.summary.FileWriter(logdir)

    env.seed(workerseed)
    np.random.seed(workerseed)
    ob_space = env.observation_space
    ac_space = env.action_space

    num_subs = args.num_subs
    macro_duration = args.macro_duration
    num_rollouts = args.num_rollouts
    warmup_time = args.warmup_time
    train_time = args.train_time


    num_batches = 15
    # observation in.
    ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[None]+list(ob_space.shape))
    # ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[None, 104])

    # features = Features(name="features", ob=ob)
    policy = Policy(name="policy", ob=ob, ac_space=ac_space, hid_size=32, num_hid_layers=2, num_subpolicies=num_subs)
    old_policy = Policy(name="old_policy", ob=ob, ac_space=ac_space, hid_size=32, num_hid_layers=2, num_subpolicies=num_subs)

    sub_policies = [SubPolicy(name="sub_policy_%i" % x, ob=ob, ac_space=ac_space, hid_size=32, num_hid_layers=2) for x in range(num_subs)]
    old_sub_policies = [SubPolicy(name="old_sub_policy_%i" % x, ob=ob, ac_space=ac_space, hid_size=32, num_hid_layers=2) for x in range(num_subs)]

    learner = Learner(env, policy, old_policy, sub_policies, old_sub_policies, comm, clip_param=0.2, entcoeff=0, optim_epochs=10, optim_stepsize=3e-5, optim_batchsize=64)
    rollout = rollouts.traj_segment_generator(policy, sub_policies, env, macro_duration, num_rollouts, stochastic=True, args=args)

    start_time = time.time()
    num_interation = 2000
    episode_rewards = {}

    for x in range(num_interation):
        callback(x)
        if x == 0:
            learner.syncSubpolicies()
            print("synced subpols")
        # Run the inner meta-episode.

        policy.reset()
        learner.syncMasterPolicies()

        env.randomizeCorrect() # change goal in this function, do not change goal in reset, make sure the logic of done-reset is correct
        # shared_goal = comm.bcast(env.single_goal, root=0)
        # env.single_goal = shared_goal
        if args.reward_level == 1:
            print("It is iteration %d so i'm changing the goal to %s" % (x, env.single_goal))
        elif args.reward_level == 2:
            print("It is iteration %d so i'm changing the goal to %s" % (x, env.realgoal))
        mini_ep = 0 if x > 0 else -1 * (rank % 10)*int(warmup_time+train_time / 10)
        # mini_ep = 0

        totalmeans = []
        while mini_ep < warmup_time+train_time:

            mini_ep += 1
            # rollout
            rolls = rollout.__next__()
            allrolls = []
            allrolls.append(rolls)
            # train theta
            rollouts.add_advantage_macro(rolls, macro_duration, 0.99, 0.98)
            gmean, lmean = learner.updateMasterPolicy(rolls)

            try:
                episode_rewards[env.single_goal] += [gmean]
            except Exception as e:
                episode_rewards[env.single_goal] = [gmean]
            # train phi
            test_seg = rollouts.prepare_allrolls(allrolls, macro_duration, 0.99, 0.98, num_subpolicies=num_subs)
            learner.updateSubPolicies(test_seg, num_batches, (mini_ep >= warmup_time))

        if rank in [0]:
            print_string = ""
            summary = tf.Summary()

            try:
                print_string += "[{}] goal {}, remaining {:.2f} hours".format(
                    x,
                    env.single_goal,
                    (time.time()-start_time)/(x)*(num_interation-x)/60.0/60.0,
                )
            except Exception as e:
                pass

            print_string += ", ep_rew for {}-th goal: {:.2f}".format(
                env.single_goal,
                episode_rewards[env.single_goal][-1],
            )

            summary.value.add(
                tag = 'ep rew for goal {}'.format(
                    env.single_goal,
                ),
                simple_value = episode_rewards[env.single_goal][-1],
            )

            summary.value.add(
                tag = 'ep rew (all) for goal {}'.format(
                    env.single_goal,
                ),
                simple_value = np.mean(episode_rewards[env.single_goal]),
            )

            print(print_string)
            summary_writer.add_summary(summary, x)
            summary_writer.flush()
            # learner.updateSubPolicies(test_seg,
            # log
            # print(("%d: global: %s, local: %s" % (mini_ep, gmean, lmean)))
            # if args.s:
            #     totalmeans.append(gmean)
            #     with open('outfile'+str(x)+'.pickle', 'wb') as fp:
            #         pickle.dump(totalmeans, fp)
