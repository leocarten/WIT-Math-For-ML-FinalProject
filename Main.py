import math

import Data
class Main:
    def __init__(self):
        pass
    def FormatDataForLinearModelPart2(self):
        data = Data.Data("Housing.csv", False)
        print("# Part 2) Fit a Linear Model Using Least Squares")
        print("Please keep in mind, this part of the exercise is to **just** create a simple linear model. This does not include polynomial model fitting, normalizing features, removing extreme outliers, etc...")
        linear_model = data.LeastSquares()
        area = linear_model[0]
        bedrooms = linear_model[1]
        bathrooms = linear_model[2]
        stories = linear_model[3]
        mainroad = linear_model[4]
        guestroom = linear_model[5]
        basement = linear_model[6]
        hotwaterheating = linear_model[7]
        airconditioning = linear_model[8]
        parking = linear_model[9]
        prefarea = linear_model[10]
        furnishingstatus = linear_model[11]
        bias = linear_model[12]
        print("### (a) Fit a Linear Model")
        print(
            f"The **linear model generated** was: "
            f"`{area}*area + "
            f"{bedrooms}*bedrooms + "
            f"{bathrooms}*bathrooms + "
            f"{stories}*stories + "
            f"{mainroad}*mainroad + "
            f"{guestroom}*guestroom + "
            f"{basement}*basement + "
            f"{hotwaterheating}*hotwaterheating + "
            f"{airconditioning}*airconditioning + "
            f"{parking}*parking + "
            f"{prefarea}*prefarea + "
            f"{furnishingstatus}*furnishingstatus + "
            f"{bias}`\n"
        )

        print("### (b) Compute Training Error")
        MSE, median, mean = data.LinearModelMSE(True, True)
        print(f"The **MSE** was: `{MSE:,}`\n")
        print(f"**RMSE** is a good way to see how \"off\" the model was per value, this is: `${round(math.sqrt(MSE), 2):,}`\n")
        print(
            f"**Linear model assessment**: Since this model does not include polynomial model fitting, normalizing features, or removing extreme outliers, "
            f"I think this model performed relatively well given `mean={mean:,.2f}` and `median={median:,.2f}`."
        )

        print("### (c) Compute Test Error")
        testingMSE, testing_median, testing_mean = data.LinearModelMSE(False, True)
        print(f"The **MSE** was: `{testingMSE:,}`\n")
        print(f"**RMSE** is a good way to see how \"off\" the model was per value, this is: `${round(math.sqrt(testingMSE), 2):,}`\n")
        print(
            f"**Linear model assessment**: Since this model does not include polynomial model fitting, normalizing features, or removing extreme outliers, "
            f"I think this model performed relatively well given `mean={testing_mean:,.2f}` and `median={testing_median:,.2f}`."
        )

    def FormatDataForPolynomialModelPart3(self):
        data = Data.Data("Housing.csv", True)
        print("# Part 3) Fit a Polynomial Model Using Least Squares")
        print("### (b) Compute Training Error")
        MSE, median, mean = data.PolynomialModelMSE(True, True)
        print(f"The **MSE** was: `{MSE:,}`\n")
        print(f"**RMSE** is a good way to see how \"off\" the model was per value, this is: `${round(math.sqrt(MSE), 2):,}`\n")
        print(
            f"I think this model performed relatively well given `mean={mean:,.2f}` and `median={median:,.2f}`."
        )

        print("### (c) Compute Test Error")
        MSE, median, mean = data.PolynomialModelMSE(False, True)
        print(f"The **MSE** was: `{MSE:,}`\n")
        print(f"**RMSE** is a good way to see how \"off\" the model was per value, this is: `${round(math.sqrt(MSE), 2):,}`\n")
        print(
            f"I think this model performed relatively well given `mean={mean:,.2f}` and `median={median:,.2f}`."
        )

        print("### (d) Additional info about how I made the Polynomial Model more accurate")
        print("Below are the things I did (or tried) to improve this model other than adding polynomial terms:")
        print("- I tried adding polynommial terms of different degrees (e.g. `^2` or `^3`) to multiple features to see how the curve of the graph was changed.")
        print("- I removed outliers by calculating z-scores of all features and removed things that were `+-2.5` standard deviations away from the mean.")
        print("- I used the `FigureOutColumnsToDropBasedOnCorrlation()` function and tried removing some features by calculating the correlations of each feature and how they relate to other features.")
        print("- I fixed how I pre-processed my data by changing how i quantified tri-nary features, e.g. the `furnished` feature. Previously it was `-1,0,1`, but I changed it to `0,1,2`.")


def main():
    main = Main()
    # main.FormatDataForLinearModelPart2()
    main.FormatDataForPolynomialModelPart3()

main()

