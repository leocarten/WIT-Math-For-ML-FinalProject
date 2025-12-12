# Leo Carten - WIT Math for ML Final Project
This contains assignment details, my thoughts, and my notes for the project. This is in reference to `FINAL-PROJECT-Math5700-project-Fall2025.pdf`.

# Project Overview
I decided to create a model that can predict housing prices based on `area,bedrooms,bathrooms,stories,mainroad,guestroom,basement,hotwaterheating,airconditioning,parking,prefarea,furnishingstatus` with information below:

| Feature              | Description                                                                 |
|----------------------|-----------------------------------------------------------------------------|
| Area                 | The total area of the house in square feet.                                 |
| Bedrooms             | The number of bedrooms in the house.                                        |
| Bathrooms            | The number of bathrooms in the house.                                       |
| Stories              | The number of stories in the house.                                         |
| Mainroad             | Whether the house is connected to the main road (Yes/No).                   |
| Guestroom            | Whether the house has a guest room (Yes/No).                                |
| Basement             | Whether the house has a basement (Yes/No).                                  |
| Hot water heating    | Whether the house has a hot water heating system (Yes/No).                  |
| Airconditioning      | Whether the house has an air conditioning system (Yes/No).                  |
| Parking              | The number of parking spaces available within the house.                    |
| Prefarea             | Whether the house is located in a preferred area (Yes/No).                  |
| Furnishing status    | The furnishing status of the house (Fully Furnished, Semi-Furnished, Unfurnished). |


This dataset is from [Kaggle](https://www.kaggle.com/datasets/harishkumardatalab/housing-price-prediction).

# Part 1) Fit a Linear Model Using Least Squares
Please keep in mind, this part of the exercise is to **just** create a simple linear model. This does not include polynomial model fitting, normalizing features, removing extreme outliers, etc...
## (a) Fit a Linear Model
The **linear model generated** was: `249.23*area + 132986.72*bedrooms + 1007369.83*bathrooms + 385471.58*stories + 442024.3*mainroad + 192035.0*guestroom + 445474.8*basement + 877903.15*hotwaterheating + 887785.09*airconditioning + 296559.85*parking + 576254.87*prefarea + 232893.99*furnishingstatus + -104625.44`

### (b) Compute Training Error
The **MSE** was: `1,111,730,738,155.23`

**RMSE** is a good way to see how "off" the model was per value, this is: `$1,054,386.43`

**Linear model assessment**: Since this model does not include polynomial model fitting, normalizing features, or removing extreme outliers, I think this model performed relatively well given `mean=4,767,705.14` and `median=4,340,000.00`.
### (c) Compute Test Error
The **MSE** was: `1,173,497,175,122.26`

**RMSE** is a good way to see how "off" the model was per value, this is: `$1,083,280.75`

**Linear model assessment**: Since this model does not include polynomial model fitting, normalizing features, or removing extreme outliers, I think this model performed relatively well given `mean=4,762,825.69` and `median=4,305,000.00`.
