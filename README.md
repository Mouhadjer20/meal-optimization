
# 🍽️ Meal Optimization Project  

This project aims to create a smart, optimized meal plan that satisfies daily nutritional needs while minimizing costs and adhering to various dietary constraints. The model distributes **N food items** across three meals: **breakfast, lunch, and dinner**. It incorporates nutritional requirements, food categories, and special recommendations to provide a balanced, feasible, and cost-effective meal plan.

---

## 🌟 Features
- **Balanced Nutrition**: Ensures daily nutritional requirements for key nutrients (e.g., calories, protein, vitamins) are met.
- **Smart Distribution**: Optimizes food allocation across three meals, considering food suitability for specific meals.
- **Cost Efficiency**: Minimizes meal plan cost while maintaining high nutritional value.
- **Category Constraints**: Incorporates diverse food categories such as drinks, principal items, and desserts in every meal.
- **Custom Penalties**: Penalizes inclusion of foods not recommended for certain meals (e.g., heavy foods for dinner).

---

## 📂 Project Files
1. **Input Files**
   - `food_data.csv`: Contains details about food items, nutritional values, categories, costs, and constraints.  
   - `not_recommended.csv`: Specifies which foods are not recommended for specific meals.

2. **Model File**
   - `optimization_model.mod`: Defines the mathematical optimization model, including variables, constraints, and the objective function.

3. **Solver Script**
   - `solver.py`: Python script to load data, run the optimization, and save the results.

4. **Output Files**
   - `meal_plan.csv`: Generated meal plan showing assigned food quantities for each meal.
   - `report.txt`: Summary of the optimization process, including cost and constraint violations.

---

## 📊 Input File Structure

### `food_data.csv`
| FoodID | FoodName       | Category | Calories | Protein | VitaminA | Cost (100g) | Max Quantity (100g) |
|--------|----------------|----------|----------|---------|----------|-------------|---------------------|
| 1      | Apple          | 1        | 52       | 0.3     | 54       | 90          | 10                  |
| 2      | Banana         | 1        | 89       | 1.1     | 64       | 65          | 10                  |
| ...    | ...            | ...      | ...      | ...     | ...      | ...         | ...                 |

### `not_recommended.csv`
| FoodID | MealID | NotRecommended |
|--------|--------|----------------|
| 1      | 1      | 0              |
| 1      | 2      | 0              |
| 1      | 3      | 0              |
|...     |...     |...             |

---

## ⚙️ Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/Mouhadjer20/meal-optimization.git
   cd meal-optimization
   ```
2. Install dependencies (Python + Optimization libraries):
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage
1. **Prepare Input Files**: Customize `food_data.csv` and `not_recommended.csv` with your data.
2. **Run the Solver**:
   ```bash
   python solver.py
   ```
3. **View Results**: Check the generated `meal_plan.csv` and `report.txt` for the optimized meal plan and summary.

---

## 🎯 Optimization Goal
Minimize:  

$$
Total\ Cost + \lambda \cdot Penalty\ for\ Not\ Recommended\ Foods
$$

Subject to:
- Daily nutritional requirements.
- Meal-specific food constraints.
- Category diversity in each meal.

---

## 🛠️ Built With
- **Python**: For data processing and scripting.
- **AMPL/PuLP/Gurobi**: To define and solve the optimization problem.
- **CSV**: For input/output data handling.

---

## 🤝 Contributing
Contributions are welcome! If you have suggestions or improvements, feel free to open an issue or submit a pull request.

---

## 📜 License
This project is licensed under the [MIT License](LICENSE).

---

## 💡 Inspiration
Inspired by the need for smarter meal planning in busy lifestyles, this project combines the power of optimization with nutritional science to create meaningful, personalized solutions.[Practical Work Sheet n.2](dl tp2 RO ing diet local search.pdf) 🌱

--- 

Happy Optimizing! 🚀
