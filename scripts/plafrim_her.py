#!/usr/bin/python
# -*- coding: utf-8 -*-

import datetime
import itertools

PATH_TO_RESULTS = "/projets/flowers/adrien/baselines/results/her/"  # plafrim
PATH_TO_SCRIPT = "/projets/flowers/adrien/baselines/baselines/her/experiment/train.py"  # plafrim
PATH_TO_INTERPRETER = "/home/alaversa/anaconda3/envs/py-3.6cpu/bin/python"  # plafrim

# PATH_TO_SCRIPT = "/Users/adrien/Documents/post-doc/baselines/baselines/her/experiment/train.py"  # MacBook 15"
# PATH_TO_INTERPRETER = "/usr/local/bin/python3"  # MacBook 15"
# PATH_TO_RESULTS = "/Users/adrien/Documents/post-doc/baselines/results/"  # MacBook 15"

envs = ['ArmBall-v0', 'ArmBallDense-v0']
seeds = list(range(0, 5))
replay_strategies = ['future', 'none']
n_cpu = 1
epochs = int(1000 / n_cpu)

params_iterator = list(itertools.product(envs, replay_strategies, seeds))

job_duration = datetime.timedelta(hours=4)
batch_duration = job_duration  # * nb_runs
minutes, seconds = divmod(batch_duration.total_seconds(), 60)
hours, minutes = divmod(minutes, 60)
duration_string = "{:02.0f}:{:02.0f}:{:02.0f}".format(hours, minutes, seconds)

filename = f'HER_ncpu_{n_cpu}_{seeds[0]}_{seeds[-1]}.sh'
with open(filename, 'w') as f:
    f.write('#!/bin/sh\n')
    f.write('#SBATCH -N 1\n')
    f.write('#SBATCH --exclusive\n')
    f.write('#SBATCH --mincpus 20\n')
    f.write('#SBATCH --partition=court\n')
    f.write('#SBATCH --time={}\n'.format(duration_string))
    f.write('#SBATCH --error={}.err\n'.format(filename))
    f.write('#SBATCH --output={}.out\n'.format(filename))
    f.write('rm log.txt; \n')
    f.write("export EXP_INTERP='%s' ;\n" % PATH_TO_INTERPRETER)
    for (env, replay_strategy, seed) in params_iterator:
        date = '$(date "+%d%m%y-%H%M-%3N")'
        name = f"{env}/{replay_strategy}/{date}"
        logdir = PATH_TO_RESULTS + name
        f.write('echo "=================> %s";\n' % name)
        f.write('echo "=================> %s" >> log.txt;\n' % name)
        f.write(f"$EXP_INTERP {PATH_TO_SCRIPT} --env={env} --replay_strategy={replay_strategy} --n_epochs={epochs}"
                f" --seed={seed} --num_cpu={n_cpu} --logdir={logdir} || (echo 'FAILURE' && echo 'FAILURE' >> log.txt) &\n")
        f.write('wait\n')
