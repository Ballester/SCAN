python train.py \
--data_path /opt/datasets/data/ \
--data_name coco_precomp \
--adapt_data coco_precomp \
--adapt_split val \
--adapt_batch_size 1 \
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
--model_name runs/coco_f30k_i2t_baseline/ \
--logger_name runs/coco_f30k_i2t_baseline/ \
--consistency_weight 0.0 \

python train.py \
--data_path /opt/datasets/data/ \
--data_name coco_precomp \
--adapt_data coco_precomp \
--adapt_split val \
--adapt_batch_size 1 \
--val_data f30k_precomp \
--val_split val \
--max_violation \
--bi_gru \
--agg_func=Mean \
--cross_attn=t2i \
--lambda_softmax=9 \
--num_epochs=20 \
--lr_update=10 \
--learning_rate=.0005 \
--model_name runs/coco_f30k_t2i_baseline/ \
--logger_name runs/coco_f30k_t2i_baseline/ \
--consistency_weight 0.0 \
