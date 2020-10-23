# PHICON

## 1.Introduction
This repository contains source code for paper "PHICON: Improving Generalization of Clinical Text De-identification Models via Data Augmentation"(accepted by EMNLP'20 The 3rd Clinical Natural Language Processing Workshop). PHICON is a simple yet effective data augmentation method to alleviate the generalization issue in de-identification. PHICON consists of PHI augmentation and Context augmentation (as shown in Figure 1), which creates augmented training corpora by replacing PHI entities with named-entities sampled from external sources, and by changing background context with synonym replacement or random word insertion, respectively.

<img src="PHICON_example.png" width="50%"/>
Figure. 1: Toy examples of our PHICON data augmentation. SR: synonym replacement. RI: random insertion.

## 2. Usage
### Setup
+ Download *Stanford Parser*, and change the corresponding path in rule_modules.py file
+ Install *spaCy* package


### PHI Augmentation

The i2b2 2006 and i2b2 2014 de-identification dataset can be accessed from:  https://portal.dbmi.hms.harvard.edu.

The data processing mainly refers to the guidance from:  
https://github.com/juand-r/entity-recognition-datasets/tree/master/data/i2b2_2006  
https://github.com/juand-r/entity-recognition-datasets/tree/master/data/i2b2_2014

We also show detailed steps on data process and PHI augmentation in the following two files:  
`PHI augmentation-i2b2-2006 dataset.ipynb`  
`PHI augmentation-i2b2-2014 dataset.ipynb`

If users already have de-identification datasets in BIO format, users can directly conduct PHI Augmentation according to the guidance in this file:  
`PHI augmentation-your-own-dataset.ipynb`


### Context Augmentation

`python Context_Aug.py`


## 3. Citation
Please kindly cite the paper if you use the code or any resources in this repo:
```bib
@inproceedings{yue2020phicon,
 title={PHICON: Improving Generalization of Clinical Text De-identification Models via Data Augmentation},
 author={Xiang Yue and Shuang Zhou},
 booktitle={Proceedings of the 3rd Clinical Natural Language Processing Workshop},
 year={2020}
}
```


