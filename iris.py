#I imported the essential libraries
#the dataset i took here to train is a iris flower dataset using the library scikitlearn
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
import streamlit as st
#I loaded the data set here
iris=load_iris()
df=pd.DataFrame(iris.data,columns=iris.feature_names)
df['species']=iris.target
#this is the data preprocessing step
X=df.drop('species',axis=1)
y=df['species']
# Split dataset
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
#this is the featuring engineering step
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
#I trained Random Forest model hereand why i used classifier here is becuz the out is not a continuous variable.
model=RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(X_train, y_train)
#used streamlit to build web interface
st.title("Iris Flower Species Predictor")
st.write("Enter the flower measurements below to predict the species:")
#these lines are used to buiuld input sliders in web interface to take input from users
sepal_length=st.slider("Sepal Length (cm)",float(df['sepal length (cm)'].min()),float(df['sepal length (cm)'].max()),float(df['sepal length (cm)'].mean()))
sepal_width=st.slider("Sepal Width (cm)",float(df['sepal width (cm)'].min()),float(df['sepal width (cm)'].max()),float(df['sepal width (cm)'].mean()))
petal_length=st.slider("Petal Length (cm)",float(df['petal length (cm)'].min()),float(df['petal length (cm)'].max()),float(df['petal length (cm)'].mean()))
petal_width =st.slider("Petal Width (cm)",float(df['petal width (cm)'].min()),float(df['petal width (cm)'].max()),float(df['petal width (cm)'].mean()))
#this is to build predict button in web page
if st.button("Predict Species"):
    input_data=np.array([[sepal_length,sepal_width,petal_length,petal_width]])
    input_scaled=scaler.transform(input_data)
    prediction=model.predict(input_scaled)
    predicted_species=iris.target_names[prediction][0]
    st.success(f"Predicted Species:🌼{predicted_species}")
#here this code is to add extra features like showing the data set i used and visualisations and accuracy score,confusion matrix,visualising the graphs distribution using pairplot
if st.checkbox("Show Dataset"):
    st.dataframe(df)
if st.checkbox("Show Pairplot"):
    st.write("Pairplot by species")
    pairplot_fig=sns.pairplot(df,hue='species')
    st.pyplot(pairplot_fig.fig)
if st.checkbox("Show Confusion Matrix"):
    y_pred=model.predict(X_test)
    cm=confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm,annot=True,cmap='Blues',fmt='d')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(plt.gcf())
#this section is used to show how each feature of the flower that contributed to predict the output
if st.checkbox("Show Feature Importance"):
    st.write("Feature importance indicates how much each feature contributed to the model prediction.")
    importances=model.feature_importances_
    feature_names=df.columns[:-1]
    fig,ax=plt.subplots()
    sns.barplot(x=importances,y=feature_names,palette="viridis",ax=ax)
    ax.set_title("Feature Importance")
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("Features")
    st.pyplot(fig)

