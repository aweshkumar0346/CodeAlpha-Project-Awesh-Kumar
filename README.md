# CodeAlpha-Project-Diease Prediction from Medical Data
I have developed a comprehensive machine learning project aimed at predicting the likelihood of diseases based on a dataset containing medical records. The project utilizes Python's powerful data science libraries, including pandas, numpy, scikit-learn, matplotlib, seaborn, and Flask. Initially, the dataset was preprocessed by handling missing values, encoding categorical variables, and standardizing numerical features. The data was then split into training and testing sets to build and evaluate the predictive models.

Two primary models, a Random Forest Classifier and a Logistic Regression model, were trained on the dataset. The performance of these models was assessed using metrics like accuracy and classification reports. To further enhance the Random Forest model's performance, hyperparameter tuning was conducted using GridSearchCV, resulting in an optimized model. The best-performing model was saved for future use, ensuring that the most accurate predictions could be made.

To make the predictive model accessible, I created a Flask web application. This application serves as an API endpoint, allowing users to send medical data in JSON format and receive disease predictions. The Flask app loads the pre-trained model and processes incoming data, ensuring it matches the format used during training. The app then returns predictions, making it a practical tool for real-world applications where quick and accurate medical predictions are needed.

To test the Flask API, I provided instructions for using Postman, a tool for sending HTTP requests, and for creating a simple HTML form that can be opened in a web browser. These methods enable users to interact with the API by submitting medical data and receiving predictions. This project not only demonstrates the application of machine learning in healthcare but also showcases the integration of a machine learning model into a web application, making advanced predictive analytics accessible to end-users.






