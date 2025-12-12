from statistics import median

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sympy
from sympy import Matrix

class Data:
    def __init__(self, filePath):
        self.filePath = filePath
        self.csvData = pd.read_csv(self.filePath)              # data frame for ENTIRE csv file
        self.FormatData()                                      # format the data to remove string values (e.g. "No" and "Yes" becomes 0 and 1 respectively)
        self.X = self.csvData.drop('price', axis=1)     # model inputs
        self.y = self.csvData['price']                         # model outputs
        self.trainingData = self.X.sample(frac=0.8,random_state=0)    # training data
        self.testingData = self.X.drop(self.trainingData.index)              # testing data

    def FormatData(self):
        """
        We must clean the CSV data since it has string variables like "yes" and "no".

        :param dataFrame: the data frame we want to format
        :return: None
        """
        # binary yes/no columns in our data
        binary_cols = ["mainroad", "guestroom", "basement",
                       "hotwaterheating", "airconditioning", "prefarea"]

        # replace "no" with 0 and "yes" with 1
        for col in binary_cols:
            self.csvData[col] = self.csvData[col].map({"no": 0, "yes": 1}).astype(int)

        # furnishing status column since it is trinary
        self.csvData["furnishingstatus"] = self.csvData["furnishingstatus"].map({
            "unfurnished": -1,
            "semi-furnished": 0,
            "furnished": 1
        }).astype(int)

    def PrintData(self):
        """
        Print CSV Data
        :return: None
        """
        print("Entire Data Set:")
        print(self.csvData)

        print("Training data:")
        print(self.trainingData)

        print("Testing data:")
        print(self.testingData)

    def NormalizeData(self, dataFrame):
        """
        Normalizing your data is important because it allows for mor meaningful data
        :param dataFrame:
        :return:
        """

    def LinearModelPrediction(
            self,
            iarea,
            ibedrooms,
            ibathrooms,
            istories,
            imainroad,
            iguestroom,
            ibasement,
            ihotwaterheating,
            iairconditioning,
            iparking,
            iprefarea,
            ifurnishingstatus,
    ):
        linear_model = self.LeastSquares()
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

        y_hat = (
                iarea * area
                + ibedrooms * bedrooms
                + ibathrooms * bathrooms
                + istories * stories
                + imainroad * mainroad
                + iguestroom * guestroom
                + ibasement * basement
                + ihotwaterheating * hotwaterheating
                + iairconditioning * airconditioning
                + iparking * parking
                + iprefarea * prefarea
                + ifurnishingstatus * furnishingstatus
                + bias
        )
        return y_hat

    def LinearModelMSE(self, b_use_trainingData):

        residual = 0
        divisor = 0
        arr = []

        if b_use_trainingData:
            dataSet = self.trainingData
        else:
            dataSet = self.testingData

        for row in dataSet.itertuples():
            # get the row index of this row, and then use the index to do a row lookup on the original CSV data which contains the price
            row_index = row.Index
            master_row_with_price = self.csvData.loc[row_index]

            y_hat = self.LinearModelPrediction(
                master_row_with_price.area,
                master_row_with_price.bedrooms,
                master_row_with_price.bathrooms,
                master_row_with_price.stories,
                master_row_with_price.mainroad,
                master_row_with_price.guestroom,
                master_row_with_price.basement,
                master_row_with_price.hotwaterheating,
                master_row_with_price.airconditioning,
                master_row_with_price.parking,
                master_row_with_price.prefarea,
                master_row_with_price.furnishingstatus,
            )

            y = master_row_with_price.price
            arr.append(y)       # append the actual price so we can get the median and mean after
            y_hat = round(y_hat, 2)
            residual += ((y_hat - y) ** 2)
            divisor += 1

        median = np.median(arr)
        mean = np.mean(arr)
        return round(residual / divisor, 2), median, mean


    def LeastSquares(self):
        # remember, IRL there is no "perfect" solution - so LSE is able to find the line of best fit with a bias
        # formula = At * Ax = At * b
        A = self.trainingData.copy()
        A['bias'] = 1       # add a column called "bias" and default it to all 1s
        At = A.transpose()

        # b is the "price" matrix - but we want to make sure we only get 80% since the training data is only 80% (will make dataframes the same size)
        b = self.y.sample(frac=0.8,random_state=0)
        At_A = At @ A       # matrix multiplication syntax for a pandas dataframe
        At_b = At @ b       # matrix multiplication syntax for a pandas dataframe

        # add At_b as the "augmented column" to At_A
        combined_augumented_matrix = At_A.copy()
        combined_augumented_matrix['augmented_column'] = At_b

        # convert to matrix so we can row-reduce
        matrix = Matrix(combined_augumented_matrix.values)
        rref_matrix, pivot_columna = matrix.rref()

        # last columns are the solution
        solution_columns = rref_matrix[:, -1]

        # add each term to an array and round the float to 2 decimal values
        float_weights = [round(float(val), 2) for val in solution_columns]

        return float_weights



