# ViTCod Algorithm on DeiT

For deit, we built on the official implementation (https://github.com/facebookresearch/deit).

We have provided pretrained checkpoints of DeiT models with low rank approximations of its attention map, and DeiT models with low rank approximations and sparse attention masks, both pretrained from ImageNet 2012.
https://drive.google.com/drive/folders/1G4naJROahOznkRMj5zJPc3KSUdHvlk7Z?usp=sharing

We have provided the attention masks for DeiT models at different sparsity. To directly finetune a DeiT model with low rank approximations of its original attention map with an additional sparse attention mask, 
```
python -m torch.distributed.launch \
--nproc_per_node=8 --use_env main.py \
--model deit_base_patch16_224 \
--resume /path/to/low_rank_checkpoint \
--data-path /path/to/imagenet \
--lr 1e-5 \
--weight-decay 1e-8 \
--epochs 100 \
--min-lr 1e-5 \
--output_dir /path/to/output_dir \
--mask_path /path/to/attention_mask \
--svd_type 'mix_head_fc_qk' \
--restart_finetune
```

## Step by step instructions
Preparation steps:
Replace `vision_transformer.py` file in `timm` library with `timm/vision_transformer.py` in this repo. Add `timm/mask_utils.py` and `timm/utils.py` to `/timm/models/` in the `timm` library.

Step 1:
To finetune a pre-trained DeiT model with low rank approximation of its attention map, execute the following command
```
python -m torch.distributed.launch \
--nproc_per_node=8 --use_env main.py \
--model deit_base_patch16_224 \
--resume https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth \
--data-path /path/to/imagenet \
--lr 1e-4 \
--weight-decay 1e-8 \
--epochs 100 \
--min-lr 5e-6 \
--output_dir /path/to/lowrank_output_dir \
--svd_type 'mix_head_fc_qk'
```
To evaluate the model, run
```
python main.py \
--eval \
--model deit_base_patch16_224 \
--finetune /path/to/lowrank_model \
--data-path /path/to/imagenet \
--svd_type 'mix_head_fc_qk' \
--batch-size 256
```
the evaluation should give
```
* Acc@1 81.576 Acc@5 95.214 loss 0.854
```

Step 2:
To generate average attention maps DeiT trained with low rank approximation of its attention map (from step 1) on the training data of ImageNet 2012,
```
python main.py \
--eval \
--model deit_base_patch16_224 \
--finetune /path/to/lowrank_ckpt \
--data-path /path/to/imagenet \
--svd_type 'mix_head_fc_qk' \
--need_weight \
--output_dir /path/to/lowrank_attention_map \
--batch-size 256
```
Step 3:
To generate attention masks of various sparsity levels given the average attention map on training data,
```
python gen_mask.py \
--method 'info' \
--attn  /path/to/attention_map\
--info_cut 0.58 \
--output_dir /path/to/attention_mask'
```
In the command above, the `method` argument controls which method to generate masks. In our paper, we used `info`. The `info_cut` argument is a number between `[0,1]` that determines the sparsity level of the output attention mask (for more details, we refer the readers to the paper). 

Step 4:
Finetune a DeiT model with low rank approximations of its original attention map with an additional attention mask,
```
python -m torch.distributed.launch \
--nproc_per_node=8 --use_env main.py \
--model deit_base_patch16_224 \
--resume /path/to/low_rank_checkpoint \
--data-path /path/to/imagenet \
--lr 1e-5 \
--weight-decay 1e-8 \
--epochs 100 \
--min-lr 1e-5 \
--output_dir /path/to/output_dir \
--mask_path /path/to/attention_mask \
--svd_type 'mix_head_fc_qk' \
--restart_finetune
```
To evaluate the model, run
```
python main.py \
--eval \
--model deit_base_patch16_224 \
--resume /path/to/lowrank_model \
--data-path /path/to/imagenet \
--svd_type 'mix_head_fc_qk' \
--mask_path /path/to/attention_mask \
--batch-size 256
```
the evaluation should give
```
* Acc@1 80.720 Acc@5 94.890 loss 0.888
```


