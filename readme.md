A basic implementation for bag of features/bag of word CV version.

This code is written in jupyter notebook. Aiming at (1)easy to make changes and (2) convinient to have a shallow try on this algorithm. So I won't be adapting it into an executable file.


Using Python3
Requires CV2, numpy, sklearn.

colab link:
https://colab.research.google.com/drive/1RyMZVRwgvBK9u94sDxv3p0emZl_wZg5P?usp=sharing

Pipeline
----------------
The whole pipeline using bag-of-features to do image classification(or other downstream task, whatever) :
1. Get feature discriptor for every train image
2. Gather all discriptors and cluster them. Each cluster centre will become the "feature"/"word" and form a dictionary.
3. Translate images into bag-of-feature dictionaries. (For each discriptor in each image, use the cluster classifier trained in step2. Then count the frequency for each feature in this image.)
4. Use the classifier/regressor you like to train the data for following tasks.

TODO
----------------
Current code is a very basic version. Things could be improved:

1. Add preprocess to images. Also argumentation.
2. Add more selections for other feature discriptor(e.g. SURF,HOG)
3. Current code cannot afford large dataset since sklearn cluster is super slow. I do have a draft version using Yinyang Kmeans with GPU(https://github.com/src-d/kmcuda) but not very sure about how to merge it into the system.
4. Collect and refactor public parameters. Assign them at the beginning of notebook
