
# ===========================> Resnet 18 <=====================================

python3 -m src.mainV2 \
		-F logs/tim_gd/mini/resnet18 \
		with dataset.path="data/mini_5" \
		ckpt_path="checkpoints/mini/softmax/resnet18" \
		dataset.split_dir="split/mini_5" \
		model.arch='resnet18' \
		evaluate=True \
		tim.iter=100 \
		eval.method='tim_gd' \
		eval.target_split_dir="split/mini_5" \
		eval.checking=True \
		eval.n_ways=5 \
		eval.shots=5 \
		eval.fresh_start=False \
		