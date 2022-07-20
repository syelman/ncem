python train_node_level.py \
--encoder_hidden_dims 30 30 \
--latent_dim=30 \
--num_workers=12 \
--batch_size=128 \
--accelerator=gpu \
--lr=0.01 \
--weight_decay=0.0 \
--check_val_every_n_epoch=5 \
--log_every_n_steps=10 \
--max_epochs=1000 \
# --gradient_clip_val=0.1 \