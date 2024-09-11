CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.run \
    --nnodes 1 \
    --master_addr localhost  \
    --master_port 12345 \
    --node_rank 0 \
    --nproc_per_node 1 \
    infer.py \
    --config ./configs/infer.yaml \
    --input_path 2024-09-10_09-11-07_6467.png \
    --lmk_path 20240911_074529_mppose.npy \
    --output_path ./output/