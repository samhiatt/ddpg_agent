#!/usr/bin/env python3
import subprocess
import json
# import pandas as pd
# from collections import namedtuple

def get_offers():
    return json.loads(subprocess.check_output('vast search offers -o dph --raw'.split(' ')).decode())

def get_instances():
    return json.loads(subprocess.check_output('vast show instances --raw'.split(' ')).decode())

def get_ssh_connect_command(instance, check_host_key=False):
    return 'ssh %s root@%s -p %i'%\
        ('-o "StrictHostKeyChecking no"' if check_host_key else '',
        instance['ssh_host'],
        instance['ssh_port'])

def get_hyperopt_worker_command(instance):
    #remote_command = "PYTHONPATH=$PYTHONPATH:~/ddpg_agent hyperopt-mongo-worker --mongo localhost:27017/hyperopt"
    remote_command = "cd ddpg_agent && git pull && ./kill_hyperopt_mongo_workers.sh && ./start_hyperopt_mongo_workers.sh 4 ; /bin/bash"
    return get_ssh_connect_command(instance)+' "%s"'%remote_command

for instance in get_instances():
    if instance['actual_status']=='running':
        print(get_hyperopt_worker_command(instance))
