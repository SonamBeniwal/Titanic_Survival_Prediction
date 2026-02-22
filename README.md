Project Overview: The Titanic Survival Prediction project is a classic binary classification machine learning initiative that analyzes passenger data (age, gender, class, fare) to predict survival outcomes. Using datasets from Kaggle, it involves data cleaning, exploratory data analysis (EDA), and training models like Logistic Regression or Random Forests to achieve high prediction accuracy.

Objective: To build a predictive model determining whether a passenger survived (1) or died (0) based on features like socio-economic status, age, gender, and family size.

Methodology: 

->Data Cleaning: Handling missing values, such as filling missing ages with the median and dropping irrelevant columns like Cabin.

->Data Visualization: Utilizing tools like Matplotlib/Seaborn to analyze relationships (e.g., gender vs. survival rate).

->Modeling: Implementing algorithms such as Logistic Regression, Decision Trees, or Random Forest via libraries like scikit-learn.

Role of various field in 'Survival Prediction' :-

1. Primary Features (Highest Importance):

->Sex (Gender): This is the most critical predictor, as the "women and children first" policy was heavily enforced during the evacuation. Females had a significantly higher chance of survival (~74%) compared to males (~19%).

->Pclass (Ticket Class/Socio-economic Status): A strong indicator of survival, with 1st-class passengers having the highest priority for lifeboats, followed by 2nd, then 3rd. Lower-class cabins were also located on lower decks, making escape more difficult.

->Age: Children (especially young ones) had a higher probability of survival due to priority, while the elderly were less likely to survive. Age is often used to create categories like "child" or "adult" to improve model accuracy.

2. Secondary Features (Moderate to Low Importance):

->Fare (Ticket Price): Correlated with Pclass; passengers paying more had better, higher locations on the ship, increasing their chances.

->SibSp & Parch (Family Size): These indicate the number of siblings/spouses and parents/children aboard. Passengers with a small family size (1–3 members) had better survival rates than those traveling alone or with very large families. They are often combined into a single "Family Size" feature.

->Embarked (Port of Embarkation): Indicates where the passenger boarded (Southampton, Cherbourg, or Queenstown). While less critical, it can show correlations with survival when combined with Pclass, as passengers from certain ports (e.g., Cherbourg) had higher proportions of 1st-class tickets

3. Features Used in Feature Engineering (Transformational Role)

->Name: Contains titles (Mr, Miss, Mrs, Master, Dr, etc.) that can be extracted to identify social status or gender more accurately (e.g., distinguishing a young boy "Master" from an adult "Mr").

->Cabin: Has too many missing values to be used directly, but the presence or absence of a cabin number (or just the deck level) can be used to indicate socioeconomic status.

->Ticket: Often used to identify family members or groups traveling together to create a more accurate Family Size feature.

