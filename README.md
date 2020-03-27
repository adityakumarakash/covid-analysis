# Covid-19 Analysis
In this analysis, we learn a deep convolutional model to predict covid-19 from chest xray images. Using the learned model we try to derive some insights and find out meaningful features learnt from chest-xrays.

## Prediction task
A resnet18 model is trained for the prediction task. The xrays of positive covid patients are taken from [covid-chestxray-dataset](https://github.com/ieee8023/covid-chestxray-dataset), and is placed in data/ folder. Xrays for healthy patients is taken from [kaggle dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) and also placed in data/ folder.\
[gen_dataset.ipynb](https://github.com/adityakumarakash/covid-analysis/blob/master/gen_dataset.ipynb) creates the dataset partition files in dataset_partition folder. On the current split resnet18 gets ~93% test accuracy.

*Detailed experiment is present in the following notebook* :\
[chestxray-analysis.ipynb](https://github.com/adityakumarakash/covid-analysis/blob/master/chestray-analysis.ipynb)\
Takes few seconds to load due to images. Grad-CAM maps are included for visualization.

## MultiLabel Prediction task
A resnet18 model is trained for the multilabel prediction task. The data consists of covid-19, viral, bacterial and normal chest xrays. The viral and bacterial chest x-rays are taken from the kaggle dataset mentioned in the previous experiments.

*Detailed experiment is present in the following notebook* :\
[chestray-multilabel-analysis.ipynb](https://github.com/adityakumarakash/covid-analysis/blob/master/chestray-multilabel-analysis.ipynb)\
Takes few seconds to load due to images. Grad-CAM maps are included for visualization.

## Setting up the project
### Data
* Create data/ folder
* Download the [covid-chestxray-dataset](https://github.com/ieee8023/covid-chestxray-dataset) and [kaggle dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) in data/
* Use [gen_dataset.ipynb](https://github.com/adityakumarakash/covid-analysis/blob/master/gen_dataset.ipynb) to generate new partition or use the existing partition provided here.
