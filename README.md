![performance_table](https://user-images.githubusercontent.com/55173544/210480097-a378b009-2f9d-4610-9b22-d73240572f34.PNG)




# branch별 명령어
#### 모든 모델,데이터별 hyper-parameter는 configs/example_opt.yaml을 확인하세요. 

## 1. mask_cl : RDEMKT 성능확인 branch  
<RDEMKT>
```
CUDA_VISIBLE_DEVICES=0 python main.py --model_name cloze --data_name algebra05 --use_wandb 0
```     
    
## 2. rotary_cl : RDEKT 성능확인 branch  
<RDEKT>
```
CUDA_VISIBLE_DEVICES=0 python main.py --model_name cloze --data_name algebra05 --use_wandb 0
```     

## 3. only_rp : 다른 KT 모델들 성능 확인 branch 
```
CUDA_VISIBLE_DEVICES=0 python main.py --model_name sakt --data_name algebra05  --use_wandb 0
```
