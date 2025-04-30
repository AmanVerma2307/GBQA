#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python './Scripts/GBQA_CU_HandLogin_Trainer.py' --s_id 14 --exp_name GBQA_CU_14_HandLogin
CUDA_VISIBLE_DEVICES=0 python './Scripts/GBQA_CU_HandLogin_Tester.py' --s_id 14 --exp_name GBQA_CU_14_HandLogin

CUDA_VISIBLE_DEVICES=0 python './Scripts/GBQA_CU_HandLogin_Trainer.py' --s_id 15 --exp_name GBQA_CU_15_HandLogin
CUDA_VISIBLE_DEVICES=0 python './Scripts/GBQA_CU_HandLogin_Tester.py' --s_id 15 --exp_name GBQA_CU_15_HandLogin

python './Scripts/GBQA-CU_dataGenerator_HandLogin.py' --s_id 16
CUDA_VISIBLE_DEVICES=0 python './Scripts/GBQA_CU_HandLogin_Trainer.py' --s_id 16 --exp_name GBQA_CU_16_HandLogin
CUDA_VISIBLE_DEVICES=0 python './Scripts/GBQA_CU_HandLogin_Tester.py' --s_id 16 --exp_name GBQA_CU_16_HandLogin