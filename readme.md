# Description
This project, is a flask application aims to predict the category of an input text in german language. 
# project files
preprocess_data : contains the functions (utils) used to preprocess and clean the data
# How to use it!
use python 3.xx
i'm using python 3.8
## 1. Install the requirements file
pip install -r requirements.txt
## 2. Run the application
Flask run
## 3. Open the link below in your browser
http://127.0.0.1:5000/
# Run docker image
### 1 . first build the image
### docker build --tag <name:tag>
for example : docker build --tag flask:1.0  
### 2 . run the cotnainer with this command 
### docker run -d <name_of_thecontainer:tag>
for example : docker run -d flask:1.0
## Perspectives
1. Train deep learning models and enhance model performance
2. add language detector to the pipeline to only accept german language
### thanks !!!
