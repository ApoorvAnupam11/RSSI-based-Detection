# 1. Importing libraries
import pickle 
import pandas as pd
from sklearn.model_selection import train_test_split  
from sklearn.metrics import classification_report, confusion_matrix 

# 2. Import unknown data
wifidata = pd.read_csv("./f2.csv")

# 3. load the model from disk
filename = 'finalized_model.pkl'
model = pickle.load(open(filename,'rb'))



# 4. Making Predictions
y_pred = model.predict(wifidata)
for i in y_pred:
    print(i)


