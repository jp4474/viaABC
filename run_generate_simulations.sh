#!/bin/bash

nohup python3 generate_training_data.py > output_generate_training_data.log --train_sizes 500000 100000 100000 2>&1 &

# End of file