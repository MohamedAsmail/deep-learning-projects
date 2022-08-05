import streamlit as st
import joblib,os
import pandas as pd
import re
# load model count vectorizer
news_vectorizer=open('models/final_news_cv_vectorizer.pkl','rb')
news_cv=joblib.load(news_vectorizer)

def load_prediction_models(model_file):
    loaded_model=joblib.load(open(os.path.join(model_file),'rb'))
    return loaded_model

def remove_tags(text):
    remove=re.compile(r'')
    return re.sub(remove,'',text)
def convert_lower(text):
    return text.lower()

def get_key(val,mydict):
    for key,value in mydict.items():
        if val==value:
            return key

"""New classifier"""
st.title('News Classifier')

st.info('Predictoin with classical ML')
news_text=st.text_area("Enter your Text","Type Here")

all_ml_models=["Logistic Regression","Knearest neighbour",'Random Forest','Decision Tree','Naive bayes']

model_choice=st.selectbox('Select Model',all_ml_models)
prediction_labels={'B news':0,'T news ':1,'P news':2, 'S news':3,'E news':4}

if st.button("Predict"):
    news_text=remove_tags(news_text)
    news_text=convert_lower(news_text)
    st.text(news_text)
    vect_text=news_cv.transform([news_text]).toarray()
    
    if model_choice=="Logistic Regression":
        predictor=load_prediction_models("models/newsclassifier_Logit_model.pkl")
        prediction=predictor.predict(vect_text)
        
    elif model_choice=="Knearest neighbour":
        predictor=load_predicition_models('models/newsclassifier_KNN_model.pkl')
        prediction=predictor.predict(vect_text)
        
    elif model_choice=='Random Forest':
        predictor=load_prediction_models('models/newsclassifier_RFOREST_model.pkl')
        prediction=predictor.predict(vect_text)
    elif model_choice=='Decision Tree':
        predictor=load_prediction_models('models/newsclassifier_DT_model.pkl')
        prediction=predictor.predict(vect_text)
    elif model_choice=='Naive bayes':
        predictor=load_prediction_models('models/newsclassifier_NB_model.pkl')
        prediction=predictor.predict(vect_text)
                                        
    final_result=get_key(prediction,prediction_labels)                                    
    st.success('your new Categorized as ::{}'.format(final_result))
        
