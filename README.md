# Traffic sign recognition app
Flask self-hosted web app which uses machine learning model to recognize traffic signs.
## Data download
 You can find the data that was used to train the model [here](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign). 
## Data preprocessing
Every picture that is uploaded to the site is resized to 50x50 pixels and then normalized to fit the standards of the model. Uploading small size pictures is recommended - it makes processing faster and there's higher chcane for model to recognize sign correctly. 
## Training
I approached the problem of traffic sign pictures classification with handcrafted Convolutional Neural Network. Model reached an accuracy of 97% but there's still a room for further optimization.
## Getting started
Clone the repository first to your local machine:
``` sh
git clone https://github.com/etterathe/traff-sign-rec.git
```
The following command will install required packages according to the configuration file:
``` sh
pip install -r requirements.txt
```
## Use
To run the app you can use this command:
``` sh
py app.py
```
You can reach the server at http://127.0.0.1:3000 
