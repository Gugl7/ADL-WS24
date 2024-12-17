cd src
echo "Running evaluation"
python run_seq_labeling.py \
                            --data_dir ../data/SROIE2019 \
                            --labels ../data/SROIE2019/labels.txt \
                            --model_name_or_path  microsoft/layoutlm-base-uncased \
                            --model_type layoutlm \
                            --max_seq_length 512 \
                            --do_lower_case \
                            --do_predict \
                            --logging_steps 10 \
                            --save_steps -1 \
                            --output_dir ../model/output \
                            --overwrite_output_dir \
                            --per_gpu_train_batch_size 8 \
                            --per_gpu_eval_batch_size 8