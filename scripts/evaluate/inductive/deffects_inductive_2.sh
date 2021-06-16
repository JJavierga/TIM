
# ===========================> Resnet 18 <=====================================

python3 -m src.main_inductive \
		-F logs/tim_gd/mini/resnet18 \
		with dataset.path="data/mini_5" \
		ckpt_path="checkpoints/mini/softmax/resnet18" \
		dataset.split_dir="split/mini_2" \
		model.arch='resnet18' \
		evaluate=True \
		eval.method='tim_gd' \
		eval.target_split_dir="split/mini_2" \
		eval.checking=True \
		eval.n_ways=2 \
		eval.shots=[5] \
		eval.query_shots=[8,92] \
		eval.number_tasks=100 \
