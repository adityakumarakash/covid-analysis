#Simple Covid-19 Analysis
In this repository the task of predicting covid-19 from chest xray images is tackled. We also try to get insights from the learned model to see if anything meaningful is learned for this kind of dataset.

## chestxray-analysis
In this I train a resnet18 model for the classification task of predicting covid-19 using the xrays from repositorty https://github.com/ieee8023/covid-chestxray-dataset: and normal xrays from kaggle chest xray pneumonia dataset.
chestxray-analysis.ipynb is the relevant notebook.
gen_dataset.ipynb is for generation of dataset partition files which are already present in dataset_partition.