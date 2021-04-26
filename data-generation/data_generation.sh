#!/bin/bash
trap "exit" INT TERM ERR
trap "kill 0" EXIT
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.6-py3.5-linux-x86_64.egg           # 0.9.6
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.6-py2.7-linux-x86_64.egg           # 0.9.6
python3 data-generation.py
