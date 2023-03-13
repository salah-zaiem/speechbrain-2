#!/bin/zsh

hub='facebook/hubert-large-ll60k'
num_layers='25'
encoder_dim='1024'
output_folder='/path/to/output'

declare -a ConsideredTasks=('LibriSpeechASR' 'IEMOCAP')
declare -a DownStreams=('BiLSTM' 'ecapa_tdnn')
for i in "${!ConsideredTasks[@]}"; do
	task=${ConsideredTasks[i]}
	downstream=${DownStreams[i]}
	python $task/$downstream/train.py --num_layers $num_layers --hub $hub --encoder_dim $encoder_dim --output_folder $output_folder/$task/$downstream
done


