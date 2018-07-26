#!/usr/bin/python
# -*- coding: utf-8 -*-

import datetime

PATH_TO_RESULTS = "/projets/flowers/adrien/baselines/results/ddpg/"  # plafrim
PATH_TO_SCRIPT = "/projets/flowers/adrien/baselines/baselines/ddpg/main.py"  # plafrim
PATH_TO_INTERPRETER = "/home/alaversa/anaconda3/envs/py-3.6/bin/python"  # plafrim

# PATH_TO_RESULTS = "/Users/adrien/Documents/post-doc/baselines/results/test/ddpg/"  # MacBook 15"
# PATH_TO_SCRIPT = "/Users/adrien/Documents/post-doc/baselines/baselines/ddpg/main.py"  # MacBook 15"
# PATH_TO_INTERPRETER = "/usr/local/bin/python3"  # MacBook 15"

env = 'ArmBall-v1'
seeds = list(range(0, 4))
epochs = 500
n_eval_steps = 1000

job_duration = datetime.timedelta(hours=4)
batch_duration = job_duration  # * nb_runs
minutes, seconds = divmod(batch_duration.total_seconds(), 60)
hours, minutes = divmod(minutes, 60)
duration_string = "{:02.0f}:{:02.0f}:{:02.0f}".format(hours, minutes, seconds)

filename = f'DDPG_env_{env}_trials_{seeds[0]}_{seeds[-1]}.sh'
with open(filename, 'w') as f:
    f.write('#!/bin/sh\n')
    f.write('#SBATCH -N 1\n')
    f.write('#SBATCH --exclusive\n')
    f.write('#SBATCH --partition=court_sirocco\n')
    f.write('#SBATCH --time={}\n'.format(duration_string))
    f.write('#SBATCH --error={}.err\n'.format(filename))
    f.write('#SBATCH --output={}.out\n'.format(filename))
    f.write('rm log.txt; \n')
    f.write("export EXP_INTERP='%s' ;\n" % PATH_TO_INTERPRETER)
    f.write('ngpu="$(nvidia-smi -L | tee /dev/stderr | wc -l)"\n')
    f.write('agpu=0\n')
    for seed in seeds:
        name = "DDPG_env:{}_seed:{}_date:{}".format(env, seed, '$(date "+%d%m%y-%H%M-%3N")')
        logdir = PATH_TO_RESULTS
        f.write('echo "=================> %s";\n' % name)
        f.write('echo "=================> %s" >> log.txt;\n' % name)
        f.write('export CUDA_VISIBLE_DEVICES=$agpu\n')
        f.write(f"$EXP_INTERP {PATH_TO_SCRIPT} --env-id={env} --seed={seed} --nb-epochs={epochs} --logdir={logdir}"
                f" --evaluation --nb-eval-steps={n_eval_steps} || (echo 'FAILURE' && echo 'FAILURE' >> log.txt) &\n")
        f.write("agpu=$(((agpu+1)%ngpu))\n")
    f.write('wait\n')
