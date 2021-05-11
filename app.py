import streamlit as st 
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
st.set_option('deprecation.showfileUploaderEncoding', False)
# Load the pickled model
model = pickle.load(open('SVMclassifier.pkl', 'rb')) 
# Feature Scaling
dataset = pd.read_csv('tarun PGI18CS043 - Classification Dataset2.csv')
# Extracting independent variable:
X=dataset[['Gender', 'Glucose', 'BP', 'SkinThickness', 'Insulin', 'BMI','PedigreeFunction', 'Age',]]
y=dataset['Outcome']
# Encoding the Independent Variable# Encoding Categorical data:
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X["Gender"] = labelencoder_X.fit_transform(X["Gender"])

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
def predict_disease(Age, Gender, Gulcose, BP, SkinThickness, Insulin, BMI, PedigreeFunction):
  output= model.predict(sc.transform([[Age, Gender, Gulcose, BP, SkinThickness, Insulin, BMI, PedigreeFunction]]))
  print("Patient don't have any disease:", output)
  if output==[1]:
    prediction="Patient have a disease"
  else:
    prediction="Patient don't have any disease"
  print(prediction)
  return prediction
def main():
    
    html_temp = """
   <div class="" style="background-color:Brown;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:25px;color:black;margin-top:10px;"Machine Learning Lab Experiment</p></center> 
   <center><p style="font-size:25px;color:black;margin-top:10px;"Disease Prediction using Support Vector Machine</p></center> 
   </div>
   </div>
   </div>
   """
    st.markdown(html_temp,unsafe_allow_html=True)
    st.header("Disease Prediction using Support Vector Machine")
    
    #Gender1 = st.select_slider('Select a Gender Male:1 Female:0',options=['1', '0'])
    Gender= st.number_input('Insert Gender Male:1 Female:0')
    Age = st.number_input('Insert a Age',18,60)
   
    Gulcose = st.number_input("Insert Gulcose", 9, 40)
    BP = st.number_input("Insert BP", 9, 40)
    SkinThickness = st.number_input("Insert SkinThickness", 9, 40)
    Insulin = st.number_input("Insert Insulin", 9, 50)
    BMI = st.number_input("Insert BMI", 9, 40)
    PedigreeFunction = st.number_input("Insert PedigreeFunction", 9, 40)
    resul=""
    if st.button("Predict"):
      result=predict_disease(Age, Gender, Gulcose, BP, SkinThickness, Insulin, BMI, PedigreeFunction)
      st.success('Model has predicted {}'.format(result))
      
    if st.button("About"):
      st.subheader("Developed by Tarun Kumar")
      st.subheader("Student of Poornima Group of Institutions, Jaipur")

if __name__=='__main__':
  main()
