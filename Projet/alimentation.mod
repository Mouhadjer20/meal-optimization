/*
#Cette partie, qui correspond à la solution du TP, ne concerne pas les échanges.

set FOOD;

param Calories { FOOD };
param Proteins { FOOD };
param Calcium { FOOD };
param VitaminA { FOOD };
param Cost { FOOD };

param dailyCalories;
param dailyProteins;
param dailyCalcium;
param dailyVitaminA;

var Select { FOOD } >= 0;

minimize cost: 
	sum {j in FOOD} Select[j] * Cost[j];
subj to calories: 
	sum {j in FOOD} Select[j] * Calories[j] >= dailyCalories;
subj to proteins: 
	sum {j in FOOD} Select[j] * Proteins[j] >= dailyProteins;
subj to calcium: 
	sum {j in FOOD} Select[j] * Calcium[j] >= dailyCalcium;
subj to vitaminA: 
	sum {j in FOOD} Select[j] * VitaminA[j] >= dailyVitaminA;
*/

/*
Et cette partie, je l'ai modifiée pour créer un solution personnel (original).

Rq: 
	-Ouvrir le fichier 'alimentation.dat' pour comprendre la modification.
*/
set BREAKFAST;

param Calories  { BREAKFAST };
param Risk 		{ BREAKFAST };
param Satisfied	{ BREAKFAST };
param Cost		{ BREAKFAST };

param dailyCalories;
param maximumRisk;
param dailySatisfied;

var Select { BREAKFAST } binary;#Avec cette variable, on peut choisir ce 'breakfast' ou pas.

minimize cost:
	sum { j in BREAKFAST } Select[j] * Cost[j];
	
subj to calories:
	sum { calorie in BREAKFAST } Select[calorie] * Calories[calorie] >= dailyCalories;
	
subj to risks:
	sum { risk in BREAKFAST } Select[risk] * Risk[risk] <= maximumRisk;
	
subj to loveIt:
	sum { j in BREAKFAST } Select[j] * Satisfied[j] >= dailySatisfied;