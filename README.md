# Covid-19 Analysis
In this analysis, we learn a deep convolutional model to predict covid-19 from chest xray images. Using the learned model we try to derive some insights and find out meaningful features learnt from chest-xrays.

## Prediction task
A resnet18 model is trained for the prediction task. The xrays of positive covid patients are taken from [covid-chestxray-dataset](https://github.com/ieee8023/covid-chestxray-dataset), and is placed in data/ folder. Xrays for healthy patients is taken from [kaggle dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) and also placed in data/ folder.\
[gen_dataset.ipynb](https://github.com/adityakumarakash/covid-analysis/blob/master/gen_dataset.ipynb) creates the dataset partition files in dataset_partition folder.

Detailed experiment is present in the following notebook :\
[chestxray-analysis.ipynb](https://github.com/adityakumarakash/covid-analysis/blob/master/chestray-analysis.ipynb)
