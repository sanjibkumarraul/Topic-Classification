# install Anaconda 

# creates a environment named test_new and installs all project specific packages
conda env create -f environment.yml

# activates the newly created test_new environment
conda activate test_new

# runs the python file as a script in test_new environment
python filename1.py

python filename2.py

python filename3.py
