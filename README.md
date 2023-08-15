# CvtGNet

## quicker start

- python -m torch.distributed.run --nproc_per_node 4 --master_port 10005 main.py --cfg configs/cvt-21-224x224.yaml --batch-size 32 --output logger  > log.txt
