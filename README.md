![performance_table](https://user-images.githubusercontent.com/55173544/210480097-a378b009-2f9d-4610-9b22-d73240572f34.PNG)





#### 모든 모델,데이터별 hyper-parameter는 configs/example_opt.yaml을 확인하세요. 

## 1.  제안 모델 성능확인  
```
#RDEMKT
CUDA_VISIBLE_DEVICES=0 python main.py --model_name rdemkt --data_name algebra05 --use_wandb 0
#RDEKT
CUDA_VISIBLE_DEVICES=0 python main.py --model_name rdemkt --data_name algebra05 --use_wandb 0 --only_rp 1
```     

## 2. 다른 KT 모델들 성능 확인 
```
CUDA_VISIBLE_DEVICES=0 python main.py --model_name sakt --data_name algebra05  --use_wandb 0
CUDA_VISIBLE_DEVICES=0 python main.py --model_name akt --data_name algebra05  --use_wandb 0
CUDA_VISIBLE_DEVICES=0 python main.py --model_name saint --data_name algebra05  --use_wandb 0
CUDA_VISIBLE_DEVICES=0 python main.py --model_name cl4kt --data_name algebra05  --use_wandb 0
```
