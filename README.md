# Topic-Classification
1. This is a GitHub link to access the replication package of the published work titled "Topic Classification using Regularized Variable-Size CNN and Dynamic BPSO in Online Social Network" in  "Arabian Journal for Science and Engineering, Springer."
2. To cite this work, please use the following citation
Raul, S.K., Rout, R.R. & Somayajulu, D.V.L.N. Topic Classification Using Regularized Variable-Size CNN and Dynamic BPSO in Online Social Network. Arab J Sci Eng (2023). https://doi.org/10.1007/s13369-023-08021-2

# Software packages Required

1. Ubuntu opearating system.

2. Pretrained fastText word embedding (*wiki-news-300d-1M.vec*) for finding word vectors can be downloaded
  https://fasttext.cc/docs/en/english-vectors.html
  
3. All the required software packages and their versions are present in the *environment.yml* file. It is a configuration file that defines a program's or application's settings.

# Hardware System Configuration

For the experimental analysis, we have used the following system configuration:

Lenovo Intel® Xeon(R) Silver 4114 CPU @ 2.20GHz × 40, NVIDIA 32GB GPU

# Dataset Link

Our Crawled Twitter dataset can be downloaded in the link: https://figshare.com/s/900ba41aaf320e228c95

However, one can run the below *SR_Twitter_Tweets_Preprocessing.py* program to obtain the pre-processed data from the original crawled Twitter data.

# Dataset Labeling

Our Crawled Twitter dataset can be labled using the python file: *Labelling_SRTwitter_Dataset.py*

# Preprocessing the Crawled Twitter Dataset

The preprocessing of Crawled Twitter Dataset can be done by using the python file: *SR_Twitter_Tweets_Preprocessing.py*

# Instructions to Run the Proposed Framework

1. Keep all the required files in a folder i.e., Datasets, pretrained fastText word embedding (*wiki-news-300d-1M.vec*), python file, configuration        file (*environment.yml*).

2. All the absolute paths of the program should be set according to the system used.

3. Open the terminal inside the folder containg all required files.

4. Install Anaconda.

5. Creates an environment named *test_new* and install all project-specific packages by typing:
  $conda env create -f environment.yml
  
6. Activates the newly created *test_new* environment by typing:
  $conda activate test_new
  
7. Runs the python file as a script in *test_new* environment:
  $python filename.py
