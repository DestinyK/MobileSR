#python main.py --model RCAN --scale 2 --save RCAN_x2 --n_resblocks 10 --n_feats 64  --patch_size 96 --ext sep --data_train DIV2K --data_test DIV2K  --epochs 1000
#python main.py --model SHUFFLENET__SR --scale 2 --save SHUFFLENET__SR_x2 --n_resblocks 16 --n_feats 256  --res_scale 0.1  --ext sep --data_train DIV2K --data_test DIV2K  --epochs 300 --save_results
#python main.py --model EMSRPLUS --scale 4 --save EMSRPLUS1_x4_groups --loss '1*L1' --n_resblocks 32 --n_feats 256  --res_scale 0.1  --ext sep --data_train DIV2K --data_test DIV2K  --epochs 300 
python main.py --model EMSRPLUS --loss '1*Mse' --n_resblocks 32 --n_feats 256 --data_test Set5+Set14+B100+Urban100+DIV2K --data_range 801-900 --scale 4 --pre_train ../experiment/EMSRPLUS1_x4_L2/model/model_latest.pt --test_only 
#python main.py --model SHUFFLESR --n_resblocks 16 --n_feats 256 --data_test Set5+Set14+B100+Urban100+DIV2K --data_range 801-900 --scale 2 --pre_train ../experiment/SHUFFLESR_x2/model/model_latest.pt --test_only
