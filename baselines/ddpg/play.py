import numpy as np
import click
import gym
import gym_flowers
from gym.wrappers import Monitor
import tensorflow as tf
from baselines.common import set_global_seeds
import glob

@click.command()
@click.argument('path', type=str, default='/Users/adrien/Documents/post-doc/baselines/results/test/ddpg/ArmBall-v1/0/tf_save/')
@click.argument('policy_file', type=str, default='best_actor_step439900_score-17.meta')
@click.option('--env_name', type=str, default='ArmBall-v0', help='Name of OpenAI Gym environment you want to test on')
@click.option('--seed', type=int, default=0)
@click.option('--n_test_rollouts', type=int, default=10)
@click.option('--render', type=int, default=1)
def main(path, policy_file, env_name, seed, render, n_test_rollouts=2):
    set_global_seeds(seed)

    # initialize environment
    try:
        env = gym.make(env_name)
    except NotImplementedError:
        print(env_name + ' does not refer to a gym environment')

    # extract space
    max_action = env.action_space.high

    # nb_actions = env.action_space.shape[-1]
    # obs_shape = (env.observation_space.shape[0] + env.unwrapped.dim_goal,)

    # actor = Actor(nb_actions, layer_norm=layer_norm)
    # critic = Critic(layer_norm=layer_norm)
    # memory = Memory(limit=1e6, action_shape=env.action_space.shape, observation_shape=obs_shape)
    # agent = DDPG(actor, critic, memory, obs_shape, env.action_space.shape,
    #              gamma=gamma, tau=tau, normalize_returns=normalize_returns,
    #              normalize_observations=normalize_observations,
    #              batch_size=batch_size, action_noise=action_noise, param_noise=param_noise, critic_l2_reg=critic_l2_reg,
    #              actor_lr=actor_lr, critic_lr=critic_lr, enable_popart=popart, clip_norm=clip_norm,
    #              reward_scale=reward_scale)

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
            infos = [env.reset()]
            actions = []
            rewards = []
            observations = []
            while not done:
                last_obs = infos[-1]['observation'].squeeze()
                last_goal = infos[-1]['desired_goal'].squeeze()
                inpt = np.concatenate([last_obs, last_goal])
                feed_dict = {obs0: [inpt]}
                action = sess.run(actor_tf, feed_dict=feed_dict)
                action *= max_action
                actions.append(action)
                if render:
                    env.render()
                out = env.step(actions[-1].squeeze())
                observations.append(out[0]['observation'])
                rewards.append(out[1])
                done = out[2]
            successes.append(out[3]['is_success'])
            score[i] = sum(rewards)
        success_rate = np.mean(successes)
        print('Success rate = %f' % success_rate)
        print(score.max())
        print(score.min())


if __name__ == '__main__':
    main()
