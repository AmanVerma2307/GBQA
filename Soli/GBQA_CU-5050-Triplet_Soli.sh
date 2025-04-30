#!/bin/bash

python './GBQA_CU-5050-Triplet_Soli_Trainer.py' --margin 0.1 --exp_name 'GBQA_CU-5050-Triplet-pt1_Soli'
python './GBQA_CU-5050-Triplet_Soli_Tester.py' --exp_name 'GBQA_CU-5050-Triplet-pt1_Soli'

python './GBQA_CU-5050-Triplet_Soli_Trainer.py' --margin 0.25 --exp_name 'GBQA_CU-5050-Triplet-pt25_Soli'
python './GBQA_CU-5050-Triplet_Soli_Tester.py' --exp_name 'GBQA_CU-5050-Triplet-pt25_Soli'

python './GBQA_CU-5050-Triplet_Soli_Trainer.py' --margin 0.5 --exp_name 'GBQA_CU-5050-Triplet-pt5_Soli'
python './GBQA_CU-5050-Triplet_Soli_Tester.py' --exp_name 'GBQA_CU-5050-Triplet-pt5_Soli'

python './GBQA_CU-5050-Triplet_Soli_Trainer.py' --margin 0.75 --exp_name 'GBQA_CU-5050-Triplet-pt75_Soli'
python './GBQA_CU-5050-Triplet_Soli_Tester.py' --exp_name 'GBQA_CU-5050-Triplet-pt75_Soli'

python './GBQA_CU-5050-Triplet_Soli_Trainer.py' --margin 1.0 --exp_name 'GBQA_CU-5050-Triplet-1_Soli'
python './GBQA_CU-5050-Triplet_Soli_Tester.py' --exp_name 'GBQA_CU-5050-Triplet-1_Soli'

python './GBQA_CU-5050-Triplet_Soli_Trainer.py' --margin 1.5 --exp_name 'GBQA_CU-5050-Triplet-1pt5_Soli'
python './GBQA_CU-5050-Triplet_Soli_Tester.py' --exp_name 'GBQA_CU-5050-Triplet-1pt5_Soli'

python './GBQA_CU-5050-Triplet_Soli_Trainer.py' --margin 2.0 --exp_name 'GBQA_CU-5050-Triplet-2_Soli'
python './GBQA_CU-5050-Triplet_Soli_Tester.py' --exp_name 'GBQA_CU-5050-Triplet-2_Soli'

python './GBQA_CU-5050-Triplet_Soli_Trainer.py' --margin 2.5 --exp_name 'GBQA_CU-5050-Triplet-2pt5_Soli'
python './GBQA_CU-5050-Triplet_Soli_Tester.py' --exp_name 'GBQA_CU-5050-Triplet-2pt5_Soli'