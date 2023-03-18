python3 benchmark.py \
    --model ViT_Transformer \
    --seq_len 196 \
    --dim 192 \
    --heads 3

python3 benchmark.py \
    --model Linformer \
    --seq_len 196 \
    --dim 192 \
    --heads 3

python3 benchmark.py \
    --model ViT_Transformer \
    --seq_len 1024 \
    --dim 192 \
    --heads 3

python3 benchmark.py \
    --model Linformer \
    --seq_len 1024 \
    --dim 192 \
    --heads 3

python3 benchmark.py \
    --model ViT_Transformer \
    --seq_len 2048 \
    --dim 192 \
    --heads 3

python3 benchmark.py \
    --model Linformer \
    --seq_len 2048 \
    --dim 192 \
    --heads 3

python3 benchmark.py \
    --model ViT_Transformer \
    --seq_len 4096 \
    --dim 512 \
    --heads 8

python3 benchmark.py \
    --model Linformer \
    --seq_len 4096 \
    --dim 512 \
    --heads 8


python3 benchmark.py \
    --model ViT_Transformer \
    --seq_len 196 \
    --dim 192 \
    --heads 3 \
    --enable_op_profiling

python3 benchmark.py \
    --model Linformer \
    --seq_len 196 \
    --dim 192 \
    --heads 3 \
    --enable_op_profiling

python3 benchmark.py \
    --model ViT_Transformer \
    --seq_len 1024 \
    --dim 192 \
    --heads 3 \
    --enable_op_profiling

python3 benchmark.py \
    --model Linformer \
    --seq_len 1024 \
    --dim 192 \
    --heads 3 \
    --enable_op_profiling

python3 benchmark.py \
    --model ViT_Transformer \
    --seq_len 2048 \
    --dim 192 \
    --heads 3 \
    --enable_op_profiling

python3 benchmark.py \
    --model Linformer \
    --seq_len 2048 \
    --dim 192 \
    --heads 3 \
    --enable_op_profiling

python3 benchmark.py \
    --model ViT_Transformer \
    --seq_len 4096 \
    --dim 512 \
    --heads 8 \
    --enable_op_profiling

python3 benchmark.py \
    --model Linformer \
    --seq_len 4096 \
    --dim 512 \
    --heads 8 \
    --enable_op_profiling