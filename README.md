**Iris Flower Classifier**

A **Streamlit web app** that predicts the species of **Iris flowers** using a **Random Forest Machine Learning model** based on sepal and petal measurements.

This project demonstrates an **end-to-end ML workflow**:  
Data → Preprocessing → Model Training → Prediction → Visualization → Insights.
---
**Features**

- Predict **Iris species**: *Setosa, Versicolor, Virginica*  
- Interactive input sliders for flower measurements  
- Display the **dataset** in a table  
- Visualize **pairplots** to explore feature relationships  
- Display **confusion matrix** for model evaluation  
- Show **feature importance** to understand which features the model relied on  

---
**Installation & Setup**

1. **Clone the repository**
```bash
git clone https://github.com/your-username/iris-flower-classifier.git
cd iris-flower-classifier
Create a virtual environment

bash
Copy code
python -m venv venv
Activate the virtual environment

Linux/Mac:

bash
Copy code
source venv/bin/activate
Windows:

bash
Copy code
venv\Scripts\activate
Install dependencies

bash
Copy code
pip install -r requirements.txt
Run the Streamlit app

bash
Copy code
streamlit run iris_app.py
Project Structure
bash
Copy code
iris-flower-classifier/
├── iris_app.py          # Main Streamlit app
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
└── screenshots/         # Optional: folder to store app screenshots
Usage
Adjust the sliders for sepal and petal length/width.

Click Predict Species to see the predicted Iris species.

Optional checkboxes let you explore dataset, visualizations, confusion matrix, and feature importance.

Technologies Used

Python 3

Streamlit – Web app framework

Scikit-learn – Machine Learning

Pandas – Data handling

NumPy – Numerical operations

Matplotlib & Seaborn – Visualizations

Future Enhancements
Add prediction history log inside the app

Deploy the app on Streamlit Cloud or Heroku

Add more interactive visualizations for better insights
