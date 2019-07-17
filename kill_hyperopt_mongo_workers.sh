#!/bin/sh

if [ `ps uax | grep bin/hyperopt-mongo-worker | grep -v grep | wc -l` -gt 0 ]; then
        echo "Killing existing processes..."
        ps uax | grep  -e hyperopt-mongo-worker | grep -v grep | awk '{print $2}' | xargs kill -9
else 
        echo "No hyperopt-mongo-worker processes found."
fi
