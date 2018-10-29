


python train.py \
--data_name coco_precomp \
--adapt_data coco_precomp \
--adapt_split val \
--adapt_batch_size 1 \
--val_data coco_precomp \
--val_split val \
--max_violation \
--bi_gru \
--agg_func=Mean \
--cross_attn=i2t \
--lambda_softmax=4 \
--num_epochs=20 \
--lr_update=10 \
--learning_rate=.0005 \
--model_name runs/coco_ti_avg/ \
--logger_name runs/coco_ti_avg/ \
--consistency_weight 0. \


python train.py \
--data_name coco_precomp \
--adapt_data coco_precomp \
--adapt_split val \
--adapt_batch_size 1 \
--val_data coco_precomp \
--val_split val \
--bi_gru \
--agg_func=Mean \
--cross_attn=t2i \
--lambda_softmax=9 \
--num_epochs=20 \
--lr_update=10 \
--learning_rate=.0005 \
--model_name runs/coco_it_avg/ \
--logger_name runs/coco_it_avg/ \
--consistency_weight 0. \



python train.py \
--data_name coco_precomp \
--adapt_data f30k_precomp \
--adapt_split val \
--adapt_batch_size 32 \
--val_data f30k_precomp \
--val_split val \
--bi_gru \
--agg_func=Mean \
--cross_attn=t2i \
--lambda_softmax=9 \
--num_epochs=40 \
--lr_update=10 \
--learning_rate=.0005 \
--model_name runs/coco_f30k_it_avg/ \
--logger_name runs/coco_f30k_it_avg/ \
--consistency_weight 10. \
--ema_late_epoch 10 \
--consistency_rampup 20 \


python train.py \
--data_name coco_precomp \
--adapt_data f30k_precomp \
--adapt_split val \
--adapt_batch_size 32 \
--val_data f30k_precomp \
--val_split val \
--max_violation \
--bi_gru \
--agg_func=Mean \
--cross_attn=i2t \
--lambda_softmax=4 \
--num_epochs=40 \
--lr_update=10 \
--learning_rate=.0005 \
--model_name runs/coco_f30k_ti_avg/ \
--logger_name runs/coco_f30k_ti_avg/ \
--consistency_weight 10. \
--ema_late_epoch 10 \
--consistency_rampup 20 \




python train.py \
--data_name coco_precomp \
--adapt_data f30k_precomp \
--adapt_split val \
--adapt_batch_size 32 \
--val_data f30k_precomp \
--val_split val \
--bi_gru \
--agg_func=Mean \
--cross_attn=t2i \
--lambda_softmax=9 \
--num_epochs=40 \
--lr_update=10 \
--learning_rate=.0005 \
--model_name runs/coco_f30k_it_avg_c20/ \
--logger_name runs/coco_f30k_it_avg_c20/ \
--consistency_weight 20. \
--ema_late_epoch 10 \
--consistency_rampup 20 \


python train.py \
--data_name coco_precomp \
--adapt_data f30k_precomp \
--adapt_split val \
--adapt_batch_size 32 \
--val_data f30k_precomp \
--val_split val \
--max_violation \
--bi_gru \
--agg_func=Mean \
--cross_attn=i2t \
--lambda_softmax=4 \
--num_epochs=40 \
--lr_update=10 \
--learning_rate=.0005 \
--model_name runs/coco_f30k_ti_avg_c20/ \
--logger_name runs/coco_f30k_ti_avg_c20/ \
--consistency_weight 20. \
--ema_late_epoch 10 \
--consistency_rampup 20 \



python train.py \
--data_name coco_precomp \
--adapt_data f30k_precomp \
--adapt_split val \
--adapt_batch_size 32 \
--val_data f30k_precomp \
--val_split val \
--bi_gru \
--agg_func=Mean \
--cross_attn=t2i \
--lambda_softmax=9 \
--num_epochs=40 \
--lr_update=10 \
--learning_rate=.0005 \
--model_name runs/coco_f30k_it_avg_c1/ \
--logger_name runs/coco_f30k_it_avg_c1/ \
--consistency_weight 1. \
--ema_late_epoch 10 \
--consistency_rampup 20 \



python train.py \
--data_name coco_precomp \
--adapt_data f30k_precomp \
--adapt_split val \
--adapt_batch_size 32 \
--val_data f30k_precomp \
--val_split val \
--max_violation \
--bi_gru \
--agg_func=Mean \
--cross_attn=i2t \
--lambda_softmax=4 \
--num_epochs=40 \
--lr_update=10 \
--learning_rate=.0005 \
--model_name runs/coco_f30k_ti_avg_c1/ \
--logger_name runs/coco_f30k_ti_avg_c1/ \
--consistency_weight 1. \
--ema_late_epoch 10 \
--consistency_rampup 20 \
