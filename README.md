BigMart MRP Prediction (Flask Deployment)

This project trains a **Decision Tree Regressor** model to predict the **MRP (Maximum Retail Price)** of BigMart products and deploys it using **Flask** with a simple **HTML frontend**.

---

## 🚀 Project Structure

BigMart_ML_Deployment/
├─ save_model.py # Train model and save pickles (model, scaler, encoders)
├─ app.py # Flask backend
├─ requirements.txt # Dependencies
├─ templates/
│ ├─ welcome.html # Page 1: Welcome page
│ ├─ input.html # Page 2: Input form page
│ └─ output.html # Page 3: Output page (prediction)
└─ Big_mart.csv # Dataset (must be placed here or update path in save_model.py)

yaml
Copy code

---

## ⚙️ Steps to Run the Project

### 1. Clone the project
```bash
git clone <your-repo-url>
cd BigMart_ML_Deployment
2. Create a virtual environment (optional but recommended)
bash
Copy code
python -m venv venv
source venv/bin/activate     # On Linux/Mac
venv\Scripts\activate        # On Windows
3. Install dependencies
bash
Copy code
pip install -r requirements.txt
4. Train and save the model
Make sure Big_mart.csv is available in the project folder, then run:

bash
Copy code
python save_model.py
This will generate:

model.pkl

scaler.pkl

encoders.pkl

5. Run the Flask app
bash
Copy code
python app.py
By default, Flask will run at:
👉 http://127.0.0.1:5000/

🌐 Web Pages
Welcome Page (/)
Displays a welcome message and link to input form.

Input Page (/input)
Collects user inputs:

Weight

FatContent

ProductVisibility

ProductType

EstablishmentYear

OutletSize

Output Page (/predict)
Displays predicted MRP value (e.g., MRP is 86.5).

📦 Requirements
Python 3.8+

Flask

pandas

scikit-learn

numpy

(Already listed in requirements.txt)

🔮 Future Improvements
Add CSS styling with Bootstrap or custom styles.

Handle unseen categorical values more gracefully.

Deploy to cloud (Heroku, Render, AWS, etc.).

Use a more advanced regression model for better accuracy.

👩‍💻 Developed as a demo ML deployment project with Flask and HTML.

yaml
Copy code

---

Do you want me to also make a **`requirements.txt`** file content right here so you can copy-paste it directly?
