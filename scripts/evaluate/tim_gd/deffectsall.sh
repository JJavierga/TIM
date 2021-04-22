
# ===========================> Resnet 18 <=====================================

python3 -m src.mainV2 \
		-F logs/tim_gd/mini/resnet18 \
		with dataset.path="data/mini_5" \
		ckpt_path="checkpoints/mini/softmax/resnet18" \
		dataset.split_dir="split/mini_5_reduced" \
		model.arch='resnet18' \
		evaluate=True \
		tim.iter=1000 \
		eval.method='tim_gd' \