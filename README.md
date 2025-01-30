# 🍽️ Diet Optimization Project  

This project focuses on optimizing diet plans using local search and metaheuristic algorithms. The goal is to create balanced meal plans that meet nutritional requirements, minimize costs, and respect constraints such as food categories and meal-specific recommendations. The model distributes **N food items** across three meals: **breakfast, lunch, and dinner**, ensuring a balanced and feasible diet plan.

---

## 🌟 Features
- **Balanced Nutrition**: Ensures daily nutritional requirements for key nutrients (e.g., calories, protein, vitamins) are met.
- **Smart Distribution**: Optimizes food allocation across three meals, considering food suitability for specific meals.
- **Cost Efficiency**: Minimizes meal plan cost while maintaining high nutritional value.
- **Category Constraints**: Incorporates diverse food categories such as drinks, main dishes, and desserts in every meal.
- **Custom Penalties**: Penalizes inclusion of foods not recommended for certain meals (e.g., heavy foods for dinner).

---

## 📂 Project Files
1. **Input Files**
   - [`diet_data.xlsx`](diet_data.xlsx): Contains details about food items, nutritional values, categories, costs, and constraints.  
   - [`dl_tp2_RO_ing_diet_local_search.pdf`](dl_tp2_RO_ing_diet_local_search.pdf): Provides the problem definition and constraints.

2. **Model File**
   - [`optimizer.py`](optimizer.py): Implements the diet optimization model using local search and interior search algorithms.

3. **Solver Script**
   - [`solver.py`](solver.py): Python script to load data, run the optimization, and display the results.

4. **Output**
   - **Console Output**: Displays the optimized meal plan, nutritional summary, and total cost.

---

## 📊 Input File Structure

### `diet_data.xlsx`
- **Table_1**: Contains food items with their nutritional values, categories, and costs.
- **Table_2**: Contains nutritional requirements for different target groups (e.g., children, adults, diabetics).

---

## ⚙️ Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/Mouhadjer20/diet-optimization.git
   cd diet-optimization
   ```
2. Install dependencies (Python + required libraries):
   ```bash
   pip install pandas numpy
   ```
   
---

## 📂 Project Files
1. **Input Files**
   - [`diet_data.xlsx`](diet_data.xlsx): Contains details about food items, nutritional values, categories, costs, and constraints.  
   - [`dl_tp2_RO_ing_diet_local_search.pdf`](dl_tp2_RO_ing_diet_local_search.pdf): Provides the problem definition and constraints.

2. **Model File**
   - [`optimizer.py`](optimizer.py): Implements the diet optimization model using local search and interior search algorithms.

3. **Solver Script**
   - [`solver.py`](solver.py): Python script to load data, run the optimization, and display the results.

4. **Output**
   - **Console Output**: Displays the optimized meal plan, nutritional summary, and total cost.

---

## 🚀 Usage
1. **Prepare Input Files:** Ensure [`diet_data.xlsx`](diet_data.xlsx) is in the project directory.
2. **Run the Solver:**
     ```bash
     python solver.py
     ```
3. **View Results:** The optimized meal plan, nutritional summary, and total cost will be displayed in the console.

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
Inspired by the need for smarter meal planning in busy lifestyles, this project combines the power of optimization with nutritional science to create meaningful, personalized solutions.[Practical Work Sheet n.2](dl_tp2_RO_ing_diet_local_search.pdf) 🌱

--- 

Happy Optimizing! 🚀
