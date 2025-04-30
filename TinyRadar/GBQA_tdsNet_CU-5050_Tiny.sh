#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python './Scripts/GBQA_tdsNet_CU-5050_Tiny_Trainer.py' --exp_name GBQA_tdsNet_CU-5050_Tiny
CUDA_VISIBLE_DEVICES=0 python './Scripts/GBQA_tdsNet_CU-5050_Tiny_Tester.py' --exp_name GBQA_tdsNet_CU-5050_Tiny

CUDA_VISIBLE_DEVICES=0 python './Scripts/GBQA_tdsNet_CU-5050_Tiny_Trainer.py' --prune True --exp_name GBQA_tdsNet_CU-5050-Prune_Tiny
CUDA_VISIBLE_DEVICES=0 python './Scripts/GBQA_tdsNet_CU-5050_Tiny_Tester.py' --prune True --exp_name GBQA_tdsNet_CU-5050-Prune_Tiny