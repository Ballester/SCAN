
# python train.py \
# --data_path /opt/datasets/data/ \
# --data_name coco_precomp \
# --adapt_data f30k_precomp \
# --adapt_split train \
# --adapt_batch_size 32 \
# --val_data f30k_precomp \
# --val_split val \
# --max_violation \
# --bi_gru \
# --agg_func=Mean \
# --cross_attn=t2i \
# --lambda_softmax=9 \
# --num_epochs=30 \
# --lr_update=10 \
# --learning_rate=.0005 \
# --model_name runs/coco_f30k_da_it_avg_c1e-1/ \
# --logger_name runs/coco_f30k_da_it_avg_c1e-1/ \
# --consistency_weight 0.1 \
# --ema_late_epoch 10 \
# --consistency_rampup 10 \
# --noise 0.01 \
# --dropout_noise 0.1 \


# python train.py \
# --data_path /opt/datasets/data/ \
# --data_name coco_precomp \
# --adapt_data f30k_precomp \
# --adapt_split train \
# --adapt_batch_size 32 \
# --val_data f30k_precomp \
# --val_split val \
# --max_violation \
# --bi_gru \
# --agg_func=Mean \
# --cross_attn=i2t \
# --lambda_softmax=4 \
# --num_epochs=30 \
# --lr_update=10 \
# --learning_rate=.0005 \
# --model_name runs/coco_f30k_da_ti_avg_c1e-1/ \
# --logger_name runs/coco_f30k_da_ti_avg_c1e-1/ \
# --consistency_weight 0.1 \
# --ema_late_epoch 10 \
# --consistency_rampup 10 \
# --noise 0.01 \
# --dropout_noise 0.1 \



python train.py \
--data_path /opt/datasets/data/ \
--data_name coco_precomp \
--adapt_data f30k_precomp \
--adapt_split train \
--adapt_batch_size 32 \
--val_data f30k_precomp \
--val_split val \
--max_violation \
--bi_gru \
--agg_func=Mean \
--cross_attn=t2i \
--lambda_softmax=9 \
--num_epochs=30 \
--lr_update=10 \
--learning_rate=.0005 \
--model_name runs/coco_f30k_da_it_avg_c1/ \
--logger_name runs/coco_f30k_da_it_avg_c1/ \
--consistency_weight 1. \
--ema_late_epoch 10 \
--consistency_rampup 10 \
--noise 0.01 \
--dropout_noise 0.1 \


python train.py \
--data_path /opt/datasets/data/ \
--data_name coco_precomp \
--adapt_data f30k_precomp \
--adapt_split train \
--adapt_batch_size 32 \
--val_data f30k_precomp \
--val_split val \
--max_violation \
--bi_gru \
--agg_func=Mean \
--cross_attn=i2t \
--lambda_softmax=4 \
--num_epochs=30 \
--lr_update=10 \
--learning_rate=.0005 \
--model_name runs/coco_f30k_da_ti_avg_c1/ \
--logger_name runs/coco_f30k_da_ti_avg_c1/ \
--consistency_weight 1 \
--ema_late_epoch 10 \
--consistency_rampup 10 \
--noise 0.01 \
--dropout_noise 0.1 \
