# DEPRECATED, use baselines.common.plot_util instead

import os
import matplotlib.pyplot as plt
import numpy as np
import json
import seaborn as sns; sns.set()
import glob2
import argparse


def smooth_reward_curve(x, y):
    halfwidth = int(np.ceil(len(x) / 60))  # Halfwidth of our smoothing convolution
    k = halfwidth
    xsmoo = x
    ysmoo = np.convolve(y, np.ones(2 * k + 1), mode='same') / np.convolve(np.ones_like(y), np.ones(2 * k + 1),
        mode='same')
    return xsmoo, ysmoo


def load_results(file):
    if not os.path.exists(file):
        return None
    with open(file, 'r') as f:
        lines = [line for line in f]
    if len(lines) < 2:
        return None
    keys = [name.strip() for name in lines[0].split(',')]
    data = np.genfromtxt(file, delimiter=',', skip_header=1, filling_values=0.)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    assert data.ndim == 2
    assert data.shape[-1] == len(keys)
    result = {}
    for idx, key in enumerate(keys):
        result[key] = data[:, idx]
    return result


def pad(xs, value=np.nan):
    maxlen = np.max([len(x) for x in xs])

    padded_xs = []
    for x in xs:
        if x.shape[0] >= maxlen:
            padded_xs.append(x)

        padding = np.ones((maxlen - x.shape[0],) + x.shape[1:]) * value
        x_padded = np.concatenate([x, padding], axis=0)
        assert x_padded.shape[1:] == x.shape[1:]
        assert x_padded.shape[0] == maxlen
        padded_xs.append(x_padded)
    return np.array(padded_xs)


def plot_data(data):
    # Plot data.
    for env_id in sorted(data.keys()):
        print('exporting {}'.format(env_id))
        n_config = len(data[env_id])
        colors = plt.get_cmap("Paired")(np.linspace(0, 1, n_config))
        plt.clf()

        for config, color in zip(sorted(data[env_id].keys()), colors):
            xs, ys = zip(*data[env_id][config])
            xs, ys = pad(xs), pad(ys)
            assert xs.shape == ys.shape

            plt.plot(xs[0], np.nanmedian(ys, axis=0), label=config, color=color)
            plt.fill_between(xs[0], np.nanpercentile(ys, 25, axis=0), np.nanpercentile(ys, 75, axis=0), alpha=0.25, color=color)
        plt.title(env_id)
        plt.xlabel('Epoch')
        plt.ylabel('Median Success Rate')
        plt.legend()
        plt.savefig(os.path.join(args.dir, 'fig_{}.png'.format(env_id)))


def format_config(params, results, env_id):
    delete = False

    replay_strategy = params['replay_strategy']
    if replay_strategy == 'future':
        config = 'her'
        epoch = np.array(results['total/episodes'])
        config += '_' + str(params['num_cpu'])
    elif replay_strategy == 'none':
        config = 'ddpg'
        epoch = np.array(results['total/episodes'])
        config += '_' + str(params['num_cpu'])
    elif replay_strategy == 'ddpg_baselines':
        config = 'ddpg_baselines'
        epoch = np.array(results['total/episodes'])
    elif any(c in replay_strategy for c in ['RPE', 'RGE', 'MGE']):
        config = replay_strategy
        epoch = np.array(results['total/episodes'])
        if 'vae' in replay_strategy:
            if not 'vae_' in replay_strategy:
                delete = True
            # config = replay_strategy + str(params['n_modules'])

    # Make all type of obs a same environment
    if not any(c in replay_strategy for c in ['RPE', 'RGE', 'MGE']):
        if 'Dense' in env_id:
            config += '-dense'
        else:
            config += '-sparse'
    env_id = env_id.replace('Dense', '')
    if 'RGB' in env_id:
        config += '-rgb'
    env_id = env_id.replace('RGB', '')
    if 'Betavae' in env_id:
        config += '-Betavae'
    env_id = env_id.replace('Betavae', '')
    if 'Vae' in env_id:
        config += '-Vae'
    env_id = env_id.replace('Vae', '')

    return config, epoch, env_id, delete


def load_params(curr_path, env_params, mge_config, her_config):
    if any(c in curr_path for c in ['RPE', 'RGE', 'MGE']):
        with open(os.path.join(curr_path, 'config.json'), 'r') as f:
            params = json.load(f)

            if params['object_size'] != env_params['object_size']:
                return None, None
            if 'MGE' in curr_path:
                if not mge_config.items() <= params.items():
                    return None, None
            if params['environment'] == 'armball':
                env_id = 'ArmBall-v0'
            if params['environment'] == 'armballs':
                env_id = 'ArmBalls-v0'
    else:
        with open(os.path.join(curr_path, 'params.json'), 'r') as f:
            params = json.load(f)
        if params['replay_strategy'] == 'future' or params['replay_strategy'] == 'none':
            if not her_config.items() <= params.items():
                return None, None
        env_id = params['env_name']
    return params, env_id


parser = argparse.ArgumentParser()
parser.add_argument('dir', type=str)
parser.add_argument('--smooth', type=int, default=1)
if __name__ == '__main__':
    args = parser.parse_args()

    env_params = {'object_size': 0.1}
    mge_config = {'distract_noise': 0.2, 'explo_noise_sdev': 0.1}
    her_config = {'num_cpu': 4, 'replay_strategy': 'future'}

    # Load all data.
    data = {}
    paths = [os.path.abspath(os.path.join(path, '..')) for path in glob2.glob(os.path.join(args.dir, '**', 'progress.csv'))]

    for curr_path in paths:
        if not os.path.isdir(curr_path):
            continue

        results = load_results(os.path.join(curr_path, 'progress.csv'))
        if not results:
            print('Deleting {}'.format(curr_path))
            import shutil
            shutil.rmtree(curr_path)
            continue

        params, env_id = load_params(curr_path, env_params, mge_config, her_config)
        if not env_id:
            print('skipping {}'.format(curr_path))
            continue

        success_rate = np.array(results['test/success_rate'])
        # epoch = np.array(results['epoch']) + 1

        config, epoch, env_id, delete = format_config(params, results, env_id)

        if delete:
            print('Deleting {}'.format(curr_path))
            import shutil
            shutil.rmtree(curr_path)

        # Process and smooth data.
        assert success_rate.shape == epoch.shape
        x = epoch
        y = success_rate
        if args.smooth:
            x, y = smooth_reward_curve(epoch, success_rate)
        assert x.shape == y.shape

        if env_id not in data:
            data[env_id] = {}
        if config not in data[env_id]:
            data[env_id][config] = []
        data[env_id][config].append((x, y))

    plot_data(data)

