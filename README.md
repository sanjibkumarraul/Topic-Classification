# Topic-Classification
This is a GitHub link to access the replication package of the under review work titled "Topic Classification using Regularized Variable-Size CNN and Dynamic BPSO in Online Social Network"

Software packages Required:
1.Ubuntu opearating system
2.Pretrained fastText word embedding (wiki-news-300d-1M.vec) for finding word vectors can be downloaded
  https://fasttext.cc/docs/en/english-vectors.html
3.All the required software packages and its versions are present in environment.yml file. It is a configuration file which defines program or application's settings.

Instructions to Running the Program:
1.Keep all the required files in a folder i.e.,Datasets,pretrained fastText word embedding (wiki-news-300d-1M.vec),python file,configuration        file(environment.yml)
2.Open the terminal inside the folder
3.Install Anaconda
4.creates a environment named test_new and installs all project specific packages by typing
  $conda env create -f environment.yml
5.Activates the newly created test_new environment by typing
  $conda activate test_new
6.Runs the python file as a script in test_new environment
  $python filename1.py

