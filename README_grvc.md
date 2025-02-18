# GRVC README

## Introduction

 As you can read in the paper and in the main readme, this repository is used for a few shot classification task.
 You need:

- Backbone: We can use one of the pretrained ones (trained in CUB or miniImageNet) that have been pretrained for input images of size 84x84, but it can be used too for bigger images. Another option would be to train our own backbone (information about that on main readme).
- Support images: The labeled images. They act as templates.
- Query images: Set of unlabeled images that we want to classify.

 Main program has been modified to enable completely unbalanced inputs with unknown distributions.

## Versions

- V2 is used when we want to use just inference and we do not know the labels of the images.
- V4 (old) is used when we know the labels of the images used for inference and we want to check.
- Versions with inductive and transductive (newer) are the same as V4, but improved.


## Inference

 First, you have to choose the support images and query images and create ./split/whatever/query.csv and ./split/whatever/support.csv following the format established in other similar files (there is no need to specify labels for the query as they are unknown). Take into account that the number of support images per class must be the same for all classes!
 Any other data regarding configuration such as number of classes (ways), number of iterations, number of support images per class (shots), datapath to split directory and datapath to images must be filled in (old) ./scripts/evaluate/img_gd/deffectsV4.sh or (new) deffects_transductive.sh and deffects_inducitve.sh.

 *We could also change the architecture and the checkpoint in case of training the backbone ourselves.

 Finally, execute the following:

 ```(bash)
 bash scripts/evaluate/img_gd/deffects_whatever.sh
 ```

 The last modified files are deffects_transductive.sh and deffects_inducitve.sh .


### Input arguments

- n_ways: Number of classes.
- shots: Number of support images (training images) per class.
- query_shots: List of numbers with the number of images that we will introduce for each defect. They must be ordered according to the alphabetic order of the labels:
[568,127,257,13,234] means [cracks, efflorescence, moss, non_defect, painting_defect, scaling, spalling].

## Checking (when we do know labels of images)

Old version:

 Everything said before applies to this section. There are only two differences:

- You will to change the configuration file which is ./scripts/evaluate/img_gd/deffectsV4.sh. *It is V4, not V2.
- Add the number of images per class that you want to use in each evaluation step and the number of evaluation step (that will be averaged to obtained final results).

 *Remember that now query.csv must have the column of labels as we want to check if the inferred values match the real ones.

 Finally, execute the following:


 ```(bash)
 bash scripts/evaluate/img_gd/deffectsV4.sh
 ```

New version:

 Files reorganized in /tim_gd/deffects_transductive.sh and /inductive/deffects_inductive.sh.

** It is defects and not deffects, but I don't file like renaming it due to interdependencies among files.
