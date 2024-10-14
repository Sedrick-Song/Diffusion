export USER=root
export http_proxy="http://star-proxy.oa.com:3128"
export https_proxy="http://star-proxy.oa.com:3128"

work_path=/apdcephfs_cq8/private_sedricksong/Encodec_DiT
python ${work_path}/run.py \
    --world_size 8 \
    --num_epochs 5 \
    --wandb True \
    --debug False \
    --train_file /apdcephfs_cq8/private_sedricksong/data/zh_magic_data/train_file \
    --dev_file /apdcephfs_cq8/private_sedricksong/data/zh_magic_data/dev_file \
    --exp_dir /apdcephfs_cq8/private_sedricksong/exp/encodec_DiT_4 \
    --batch_size 8

:<<!
python ${work_path}/inference.py \
    --ckpt /apdcephfs_cq8/private_sedricksong/exp/encodec_DiT_2/epoch-2.pt \
    --test_file /apdcephfs_cq8/private_sedricksong/data/zh_magic_data/train_file_part \
    --unet_model_config /apdcephfs_cq8/private_sedricksong/Encodec_DiT/configs/diffusion_model_config_dit_L_4.json
!
