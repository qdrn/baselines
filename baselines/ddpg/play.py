import numpy as np
import click
import gym
import gym_flowers
from gym.wrappers import Monitor
import tensorflow as tf
from baselines.common import set_global_seeds
import glob


@click.command()
@click.argument('path', type=str, default='/Users/adrien/Documents/post-doc/baselines/results/ddpg/ArmBall-v1/0/tf_save/')
@click.option('--env_name', type=str, default='ArmBall-v1', help='Name of OpenAI Gym environment you want to test on')
@click.option('--seed', type=int, default=0)
@click.option('--n_test_rollouts', type=int, default=10)
@click.option('--render', type=int, default=1)
def main(path, env_name, seed, render, n_test_rollouts=2):
    set_global_seeds(seed)

    # initialize environment
    env = gym.make(env_name)
    max_action = env.action_space.high

    # Load policy.
    save_recording = False

    if save_recording:
        saving_vid = '/media/flowers/3C3C66F13C66A59C/data_save/gym_recording/ddpg_cheetah_drop/' + weight_file[:-5]
        env = Monitor(env, saving_vid, force=True)
    # env.directory = '/media/flowers/3C3C66F13C66A59C/data_save/gym_recording/ddp_cheetah_drop'

    with tf.Session() as sess:
        # init = tf.global_variables_initializer()
        # sess.run(init)
        policy_file = glob.glob(path + '*.meta')[0]
        saver = tf.train.import_meta_graph(policy_file)
        saver.restore(sess, tf.train.latest_checkpoint(path))
        graph = tf.get_default_graph()
        obs0 = graph.get_tensor_by_name("obs0:0")
        actor_tf = graph.get_tensor_by_name("actor/Tanh:0")

        score = np.zeros([n_test_rollouts])
        successes = []
        for i in range(n_test_rollouts):
            done = False
            obs = env.reset()
            actions = []
            rewards = []
            observations = []
            while not done:
                inpt = obs
                feed_dict = {obs0: [inpt]}
                action = sess.run(actor_tf, feed_dict=feed_dict)
                actions.append(action)
                if render:
                    env.render()
                new_obs, r, done, info = env.step(action.flatten() * max_action)
                observations.append(new_obs)
                rewards.append(r)
                obs = new_obs
            if 'is_success' in info.keys():
                successes.append(info['is_success'])
            score[i] = sum(rewards)
        success_rate = np.mean(successes)
        print('Success rate = %f' % success_rate)
        print(score.max())
        print(score.min())


if __name__ == '__main__':
    main()
