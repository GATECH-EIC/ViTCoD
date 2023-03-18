CUDA_VISIBLE_DEVICES=1 python benchmark.py \
    --model ViT_Transformer \
    --seq_len 196 \
    --dim 192 \
    --heads 3 \
    --warmup_iters 3 \
    --num_iters 30 \
    --init_batch_size 8192

CUDA_VISIBLE_DEVICES=1 python benchmark.py \
    --model Linformer \
    --seq_len 196 \
    --dim 192 \
    --heads 3 \
    --warmup_iters 3 \
    --num_iters 30 \
    --init_batch_size 8192

CUDA_VISIBLE_DEVICES=1 python benchmark.py \
    --model ViT_Transformer \
    --seq_len 1024 \
    --dim 192 \
    --heads 3 \
    --warmup_iters 3 \
    --num_iters 30 \
    --init_batch_size 8192

CUDA_VISIBLE_DEVICES=1 python benchmark.py \
    --model Linformer \
    --seq_len 1024 \
    --dim 192 \
    --heads 3 \
    --warmup_iters 3 \
    --num_iters 30 \
    --init_batch_size 8192

CUDA_VISIBLE_DEVICES=1 python benchmark.py \
    --model ViT_Transformer \
    --seq_len 2048 \
    --dim 192 \
    --heads 3 \
    --warmup_iters 3 \
    --num_iters 30 \
    --init_batch_size 8192

CUDA_VISIBLE_DEVICES=1 python benchmark.py \
    --model Linformer \
    --seq_len 2048 \
    --dim 192 \
    --heads 3 \
    --warmup_iters 3 \
    --num_iters 30 \
    --init_batch_size 8192

CUDA_VISIBLE_DEVICES=1 python benchmark.py \
    --model ViT_Transformer \
    --seq_len 4096 \
    --dim 512 \
    --heads 8 \
    --warmup_iters 3 \
    --num_iters 30 \
    --init_batch_size 8192

CUDA_VISIBLE_DEVICES=1 python benchmark.py \
    --model Linformer \
    --seq_len 4096 \
    --dim 512 \
    --heads 8 \
    --warmup_iters 3 \
    --num_iters 30 \
    --init_batch_size 8192



CUDA_VISIBLE_DEVICES=1 python benchmark.py \
    --model ViT_Transformer \
    --seq_len 196 \
    --dim 192 \
    --heads 3 \
    --warmup_iters 3 \
    --num_iters 3 \
    --init_batch_size 8192 \
    --enable_op_profiling

CUDA_VISIBLE_DEVICES=1 python benchmark.py \
    --model Linformer \
    --seq_len 196 \
    --dim 192 \
    --heads 3 \
    --warmup_iters 3 \
    --num_iters 3 \
    --init_batch_size 8192 \
    --enable_op_profiling

CUDA_VISIBLE_DEVICES=1 python benchmark.py \
    --model ViT_Transformer \
    --seq_len 1024 \
    --dim 192 \
    --heads 3 \
    --warmup_iters 3 \
    --num_iters 3 \
    --init_batch_size 8192 \
    --enable_op_profiling

CUDA_VISIBLE_DEVICES=1 python benchmark.py \
    --model Linformer \
    --seq_len 1024 \
    --dim 192 \
    --heads 3 \
    --warmup_iters 3 \
    --num_iters 3 \
    --init_batch_size 8192 \
    --enable_op_profiling

CUDA_VISIBLE_DEVICES=1 python benchmark.py \
    --model ViT_Transformer \
    --seq_len 2048 \
    --dim 192 \
    --heads 3 \
    --warmup_iters 3 \
    --num_iters 3 \
    --init_batch_size 8192 \
    --enable_op_profiling

CUDA_VISIBLE_DEVICES=1 python benchmark.py \
    --model Linformer \
    --seq_len 2048 \
    --dim 192 \
    --heads 3 \
    --warmup_iters 3 \
    --num_iters 3 \
    --init_batch_size 8192 \
    --enable_op_profiling

CUDA_VISIBLE_DEVICES=1 python benchmark.py \
    --model ViT_Transformer \
    --seq_len 4096 \
    --dim 512 \
    --heads 8 \
    --warmup_iters 3 \
    --num_iters 3 \
    --init_batch_size 8192 \
    --enable_op_profiling

CUDA_VISIBLE_DEVICES=1 python benchmark.py \
    --model Linformer \
    --seq_len 4096 \
    --dim 512 \
    --heads 8 \
    --warmup_iters 3 \
    --num_iters 3 \
    --init_batch_size 8192 \
    --enable_op_profiling