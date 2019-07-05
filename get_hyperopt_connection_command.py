#!/usr/bin/env python3
import subprocess
import json
# import pandas as pd
# from collections import namedtuple

def get_offers():
    return json.loads(subprocess.check_output('vast search offers -o dph --raw'.split(' ')).decode())

def get_instances():
    return json.loads(subprocess.check_output('vast show instances --raw'.split(' ')).decode())

def get_ssh_connect_command(instance):
    return 'ssh -o "StrictHostKeyChecking no" root@%s -p %i -R 27017:localhost:27017'%\
        (instance['ssh_host'],instance['ssh_port'])

def get_hyperopt_worker_command(instance):
    remote_command = "PYTHONPATH=$PYTHONPATH:~/ddpg_agent hyperopt-mongo-worker --mongo localhost:27017/hyperopt"
    return get_ssh_connect_command(instance)+' "%s"'%remote_command

for instance in get_instances():
    print(get_hyperopt_worker_command(instance))
