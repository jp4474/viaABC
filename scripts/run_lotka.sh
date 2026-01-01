#!/bin/bash

nohup python src/inference.py inference=lotka > lotka.log 2>&1 &

# Modify the lotka config file in configs/inference folder to change parameters as needed.