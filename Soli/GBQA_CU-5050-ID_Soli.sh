#!/bin/bash

python './GBQA_CU-5050-ID_Soli_Trainer.py' --lambda_id 0.1 --exp_name 'GBQA_CU-5050-ID-pt1_Soli'
python './GBQA_CU-5050-ID_Soli_Tester.py' --lambda_id 0.1 --exp_name 'GBQA_CU-5050-ID-pt1_Soli'

python './GBQA_CU-5050-ID_Soli_Trainer.py' --lambda_id 0.25 --exp_name 'GBQA_CU-5050-ID-pt25_Soli'
python './GBQA_CU-5050-ID_Soli_Tester.py' --lambda_id 0.25 --exp_name 'GBQA_CU-5050-ID-pt25_Soli'

python './GBQA_CU-5050-ID_Soli_Trainer.py' --lambda_id 0.5 --exp_name 'GBQA_CU-5050-ID-pt5_Soli'
python './GBQA_CU-5050-ID_Soli_Tester.py' --lambda_id 0.5 --exp_name 'GBQA_CU-5050-ID-pt5_Soli'

python './GBQA_CU-5050-ID_Soli_Trainer.py' --lambda_id 1.0 --exp_name 'GBQA_CU-5050-ID-1_Soli'
python './GBQA_CU-5050-ID_Soli_Tester.py' --lambda_id 1.0 --exp_name 'GBQA_CU-5050-ID-1_Soli'

python './GBQA_CU-5050-ID_Soli_Trainer.py' --lambda_id 1.5 --exp_name 'GBQA_CU-5050-ID-1pt5_Soli'
python './GBQA_CU-5050-ID_Soli_Tester.py' --lambda_id 1.5 --exp_name 'GBQA_CU-5050-ID-1pt5_Soli'

python './GBQA_CU-5050-ID_Soli_Trainer.py' --lambda_id 2.0 --exp_name 'GBQA_CU-5050-ID-2_Soli'
python './GBQA_CU-5050-ID_Soli_Tester.py' --lambda_id 2.0 --exp_name 'GBQA_CU-5050-ID-2_Soli'