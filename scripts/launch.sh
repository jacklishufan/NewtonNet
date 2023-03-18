CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py -c config_md17_3.yml -p md17 -n 3body-0.9;
CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py -c config_md17_4.yml -p md17 -n 3body-0.2;

CUDA_VISIBLE_DEVICES=4,5,6,7 \
python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --node_rank=0 \
--master_addr="127.0.0.1" --master_port=19545 \
run_new_loader.py ;