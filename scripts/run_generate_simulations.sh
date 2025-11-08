#!/bin/bash

nohup python3 src/generate_training_data.py > output_generate_training_data.log --train_sizes 200000 20000 0 --num_workers 16 2>&1 &
echo "Training data generation started. Check output_generate_training_data.log for progress."
echo $!
# End of file