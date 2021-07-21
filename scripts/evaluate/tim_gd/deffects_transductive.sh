
# ===========================> Resnet 18 <=====================================
# Alfabetical order of classes
# All examples: eval.query_shots=[568,127,8,257,15,13,234]
#[568,127,257,13,234]
#[2,2,2,200,2,2,2]
#[2,2,200,2,2]
python3 -m src.mainV5 \
		-F logs/tim_gd/mini/resnet18 \
		with dataset.path="/media/grvc/MAMG_ViGUS/GRVC/Datasets/ML/Bridges_or_cracks/multiclassifier/MultiClassifier_csv/images_with_general_defects" \
		ckpt_path="checkpoints/mini/softmax/resnet18" \
		dataset.split_dir="/media/grvc/MAMG_ViGUS/GRVC/Datasets/ML/Bridges_or_cracks/multiclassifier/Splits/150/TIM" \
		model.arch='resnet18' \
		evaluate=True \
		tim.iter=1 \
		eval.method='tim_gd' \
		eval.target_split_dir="/media/grvc/MAMG_ViGUS/GRVC/Datasets/ML/Bridges_or_cracks/multiclassifier/Splits/150/TIM" \
		eval.checking=True \
		eval.n_ways=5 \
		eval.shots=[150] \
		eval.query_shots=[284,63,128,13,117] \
		eval.number_tasks=1000 \
		eval.fresh_start=True \