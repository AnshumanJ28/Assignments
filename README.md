# Machine Learning & Deep Learning Assignments
This repository contains a collection of Machine Learning and Deep Learning assignments completed using Python, Scikit-learn, and TensorFlow/Keras. The assignments cover regression, classification, clustering, neural networks, computer vision, and end-to-end ML deployment concepts through practical implementations.
---
## Repository Structure
```text
.
├── Assignment1.ipynb
├── ASS2.ipynb
├── ASS3.ipynb
├── ASS4.ipynb
├── ASS5.ipynb
├── ASS6.ipynb
├── ASS7.ipynb
├── ASS8.ipynb
├── ASS9.ipynb
├── HeartDiseaseDeployment/
│   ├── app.py
│   ├── train_model.py
│   ├── model.pkl
│   ├── scaler.pkl
│   ├── feature_names.pkl
│   ├── heart.csv
│   ├── requirements.txt
│   ├── Procfile
│   ├── README.md
│   ├── templates/
│   └── static/
├── README.md
```
---
## Technologies Used
- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow / Keras
- Flask
- Pillow (PIL)
- Requests
---
## Assignments
| Assignment | Topic | Algorithm |
|------------|-------|-----------|
| Assignment 1 | Medical Insurance Charges Prediction | Multiple Linear Regression |
| Assignment 2 | Telco Customer Churn Prediction | Logistic Regression |
| Assignment 3 | Employee Salary Prediction | Polynomial Regression |
| Assignment 4 | Breast Cancer Classification | K-Nearest Neighbors (KNN) |
| Assignment 5 | Employee Attrition Prediction | Decision Tree & Random Forest |
| Assignment 6 | Weather Classification | Support Vector Machine (SVM) |
| Assignment 7 | Mall Customer Segmentation | K-Means Clustering & PCA |
| Assignment 8 | Handwritten Digit Recognition | Artificial Neural Network (ANN) |
| Assignment 9 | Cats vs Dogs Image Classification | Convolutional Neural Network (CNN) |
| Heart Disease Deployment | Heart Disease Risk Prediction (deployed as a live REST API) | Random Forest + Flask + Render |
---
## Machine Learning Workflow
Each assignment follows a standard machine learning pipeline:
1. Data Collection
2. Data Understanding
3. Data Preprocessing
4. Feature Engineering (where applicable)
5. Model Development
6. Model Training
7. Model Evaluation
8. Performance Visualization
9. Conclusion

The `HeartDiseaseDeployment/` project extends this pipeline one step further into **deployment**: the trained model is served through a Flask REST API and deployed as a live web service on Render.
---
## Libraries Used
### Data Analysis
- Pandas
- NumPy
### Data Visualization
- Matplotlib
- Seaborn
### Machine Learning
- Scikit-learn
### Deep Learning
- TensorFlow
- Keras
### Deployment
- Flask
- Gunicorn
### Image Processing
- Pillow
---
## Learning Outcomes
This repository demonstrates practical implementation of:
- Multiple Linear Regression
- Logistic Regression
- Polynomial Regression
- K-Nearest Neighbors
- Decision Tree
- Random Forest
- Support Vector Machine
- K-Means Clustering
- Principal Component Analysis (PCA)
- Artificial Neural Networks (ANN)
- Convolutional Neural Networks (CNN)
It also covers:
- Data Cleaning
- Feature Engineering
- Feature Scaling
- Label Encoding
- One-Hot Encoding
- Model Evaluation
- Data Visualization
- Image Classification
- REST API Development (Flask)
- Cloud Deployment (Render)
---
## How to Run
Clone the repository:
```bash
git clone https://github.com/AnshumanJ28/Assignments.git
```
Install the required libraries:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow pillow requests
```
Run any notebook:
```bash
jupyter notebook
```
or open the desired `.ipynb` file in Jupyter Notebook, VS Code, or Google Colab.

### Running the Heart Disease Deployment project
```bash
cd Assignments/HeartDiseaseDeployment
pip install -r requirements.txt
python app.py
```
See [`HeartDiseaseDeployment/README.md`](./HeartDiseaseDeployment/README.md) for full details, API usage, and the live Render deployment URL.
---
## Repository Highlights
- 9 practical Machine Learning and Deep Learning assignments
- Covers supervised, unsupervised, and deep learning techniques
- Includes regression, classification, clustering, ANN, and CNN models
- Implements complete preprocessing and evaluation pipelines
- Includes a fully deployed ML project: Heart Disease Prediction via a live Flask REST API on Render
- Uses industry-standard Python libraries
---
## Author
**Anshuman Pandey**
Bachelor of Technology (Artificial Intelligence & Machine Learning)
GitHub: **https://github.com/AnshumanJ28**
