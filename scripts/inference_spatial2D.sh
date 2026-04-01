#!/bin/bash

nohup python src/inference.py inference=spatial2D > spatial2D.log 2>&1 &

# Modify the spatial2D config file in configs/inference folder to change parameters as needed.