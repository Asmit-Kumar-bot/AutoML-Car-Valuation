# 🚗 AutoML Car Valuation

This project uses machine learning to predict the selling price of a car based on various features using an automated ML approach in Jupyter Notebook.

## 📊 Features Used
- Year
- Present Price
- Kms Driven
- Fuel Type
- Seller Type
- Transmission
- Owner


## 🤖 Models Used

### 1. **Linear Regression**
- **Goal**: To model the relationship between the dependent variable (Selling Price) and multiple independent variables (features).
- **Mathematics**:
  - The model assumes a linear relationship:  
    ```
    y = β₀ + β₁·x₁ + β₂·x₂ + ... + βₙ·xₙ + ε
    ```
  - The coefficients (βᵢ) are estimated by minimizing the **Mean Squared Error (MSE)**:
    ```
    MSE = (1/n) * Σ(yᵢ - ŷᵢ)²
    ```

### 2. **Lasso Regression (L1 Regularization)**
- **Goal**: Same as Linear Regression, but adds regularization to reduce overfitting.
- **Mathematics**:
  - Adds a penalty term to the loss function:
    ```
    Loss = MSE + λ * Σ|βⱼ|
    ```
  - This encourages sparsity in the coefficients, effectively performing **feature selection**.
  - `λ` is the regularization parameter controlling the strength of the penalty.

---

## 📈 Evaluation Metrics

- **R² Score**: Measures how well the regression model explains the variability of the response data.
- **Mean Squared Error (MSE)**: The average squared difference between the actual and predicted values.
- **Cross-validation**: Used to evaluate model performance on unseen data.

---

## 🧠 Libraries Used
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## 📁 Files
- `AutoML_Car_Valuation.ipynb` – The main Jupyter Notebook for training and evaluating the model.
- `car_dataset.csv` – The dataset used for training the model.
- `requirements.txt` – Python dependencies required to run the notebook.

## ▶️ How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Asmit-Kumar-bot/AutoML-Car-Valuation.git
   cd AutoML-Car-Valuation

2. Install dependencies:
   ```bash
   pip install -r requirements.txt

3. Open the notebook:
   jupyter notebook AutoML_Car_Valuation.ipynb

4. Run all cells to train the model and view results.

