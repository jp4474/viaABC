#!/bin/bash

nohup python3 run_finetune.py --dirpath vae_64_2_1_2 > vae_64_2_1_2_finetune.log 2>&1 &

# End of file