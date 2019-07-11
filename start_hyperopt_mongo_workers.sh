#!/bin/sh

start_threads() {
	echo "Starting $1 hyperopt-mongo-worker threads..."
	for i in `seq 1 $1`; do 
		PYTHONPATH=$PYTHONPATH:./ddpg_agent hyperopt-mongo-worker --mongo localhost:27017/hyperopt --reserve-timeout=36000000 >> hyperopt-mongo-worker-$i.log 2>&1  & 
	done
}

cd ~/ddpg_agent && git checkout adding_quadcopter && git pull && start_threads 4
