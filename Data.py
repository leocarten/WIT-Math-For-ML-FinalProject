from statistics import median

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sympy
from sympy import Matrix
from scipy import stats

class Data:
    def __init__(self, filePath, b_remove_price_outliers):
        self.filePath = filePath
        self.csvData = pd.read_csv(self.filePath)              # data frame for ENTIRE csv file
        if b_remove_price_outliers:
            # remove price outliers
            self.RemoveOutliers()
        self.ShuffleData()
        self.FormatData()                                      # format the data to remove string values (e.g. "No" and "Yes" becomes 0 and 1 respectively)
        self.X = self.csvData.drop('price', axis=1)     # model inputs
        self.y = self.csvData['price']                         # model outputs
        self.trainingData = self.X.sample(frac=0.8,random_state=0)    # training data
        self.testingData = self.X.drop(self.trainingData.index)              # testing data

    def DisplayPriceHistogram(self):
        # Compute mean and standard deviation
        mean_price = self.csvData['price'].mean()
        std_price = self.csvData['price'].std()

        plt.figure(figsize=(8, 5))
        plt.hist(self.csvData['price'], bins=30, edgecolor='black')
        plt.title("Histogram of Prices")
        plt.xlabel("Price")
        plt.ylabel("Frequency")
        plt.grid(True)

        # Add vertical lines for mean and ±1 standard deviation
        plt.axvline(mean_price, color='red', linestyle='dashed', linewidth=2, label=f'Mean = {mean_price:.2f}')
        plt.axvline(mean_price + std_price, color='green', linestyle='dashed', linewidth=2,
                    label=f'+1 SD = {mean_price + std_price:.2f}')
        plt.axvline(mean_price + std_price, color='green', linestyle='dashed', linewidth=2,
                    label=f'+2 SD = {mean_price + 2*std_price:.2f}')
        plt.axvline(mean_price + std_price, color='green', linestyle='dashed', linewidth=2,
                    label=f'+3 SD = {mean_price + 3*std_price:.2f}')
        plt.axvline(mean_price - std_price, color='green', linestyle='dashed', linewidth=2,
                    label=f'-1 SD = {mean_price - std_price:.2f}')
        plt.axvline(mean_price - std_price, color='green', linestyle='dashed', linewidth=2,
                    label=f'-2 SD = {mean_price - 2*std_price:.2f}')
        plt.axvline(mean_price - std_price, color='green', linestyle='dashed', linewidth=2,
                    label=f'-3 SD = {mean_price - 3*std_price:.2f}')

        plt.legend()
        plt.show()

    def RemoveOutliers(self):
        # print(self.csvData)
        # Compute Z-scores for all numeric columns
        z_scores = stats.zscore(self.csvData.select_dtypes(include='number'))

        # Keep rows where all absolute Z-scores are <= 2.5
        self.csvData = self.csvData[(abs(z_scores) <= 2.5).all(axis=1)]

        # remove ALL rows where price is +- 2.25 SDs away from mean
        z_scores = stats.zscore(self.csvData['price'])
        self.csvData = self.csvData[abs(z_scores) <= 2.25]


    def ShuffleData(self):
        self.csvData = self.csvData.sample(frac=1, random_state=0)

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
            "unfurnished": 0,
            "semi-furnished": 1,
            "furnished": 2
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

    def PolynomialModelPrediction(
            self,
            # iarea,
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
        (
            # area,
            # area_squared,
            # area_cubed,
            bedrooms,
            # bedrooms_squared,
            bathrooms,
            # bathrooms_squared,
            stories,
            stories_squared,
            mainroad,
            guestroom,
            basement,
            hotwaterheating,
            airconditioning,
            parking,
            prefarea,
            furnishingstatus,
            furnishingstatus_squared,
            bias
        ) = self.LeastSquaresPolynomial()

        y_hat = (
                # iarea * area
                # + (iarea**2) * area_squared
                # + (iarea ** 3) * area_cubed
                + ibedrooms * bedrooms
                # + (ibedrooms**2) * bedrooms_squared
                + ibathrooms * bathrooms
                # + (ibathrooms**2) * bathrooms_squared
                + istories * stories
                + (istories**2) * stories_squared
                + imainroad * mainroad
                + iguestroom * guestroom
                + ibasement * basement
                + ihotwaterheating * hotwaterheating
                + iairconditioning * airconditioning
                + iparking * parking
                + iprefarea * prefarea
                + ifurnishingstatus * furnishingstatus
                + (ifurnishingstatus**3) * furnishingstatus_squared
                + bias
        )
        return y_hat

    def DisplayLineGraph(self, x_values, predicted, actual_price, graph_title):
        plt.plot(x_values, predicted, label="Predicted Price")  # First line
        plt.plot(x_values, actual_price, label="Actual Price")  # Second line
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.xlabel("House")
        plt.ylabel("Price in USD")
        plt.ticklabel_format(style='plain', axis='y')
        plt.title(graph_title)
        plt.legend()  # Show labels
        plt.show()


    def PolynomialModelMSE(self, b_use_trainingData, b_display_graph):

        residual = 0
        divisor = 0
        arr = []
        x_arr = []
        price_arr = []
        predicted_price_arr = []
        graph_title = ""

        if b_use_trainingData:
            dataSet = self.trainingData
            graph_title = "TRAINING DATA (80%) - Polynomial Model prediction vs. Actual Price"
        else:
            dataSet = self.testingData
            graph_title = "TESTING DATA (20%) - Polynomial Model prediction vs. Actual Price"


        for row in dataSet.itertuples():
            # get the row index of this row, and then use the index to do a row lookup on the original CSV data which contains the price
            row_index = row.Index
            master_row_with_price = self.csvData.loc[row_index]

            # get model prediction
            y_hat = self.PolynomialModelPrediction(
                # master_row_with_price.area,
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

            if b_display_graph:
                house_index = divisor
                x_arr.append(house_index)
                price_arr.append(y)
                predicted_price_arr.append(y_hat)

        median = np.median(arr)
        mean = np.mean(arr)

        if b_display_graph:
            self.DisplayLineGraph(x_arr, predicted_price_arr, price_arr, graph_title)

        return round(residual / divisor, 2), median, mean

    def LinearModelMSE(self, b_use_trainingData, b_display_graph):

        residual = 0
        divisor = 0
        arr = []
        x_arr = []
        price_arr = []
        predicted_price_arr = []
        graph_title = ""

        if b_use_trainingData:
            dataSet = self.trainingData
            graph_title = "TRAINING DATA (80%) - Model prediction vs. Actual Price"
        else:
            dataSet = self.testingData
            graph_title = "TESTING DATA (20%) - Model prediction vs. Actual Price"


        for row in dataSet.itertuples():
            # get the row index of this row, and then use the index to do a row lookup on the original CSV data which contains the price
            row_index = row.Index
            master_row_with_price = self.csvData.loc[row_index]

            # get model prediction
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

            if b_display_graph:
                house_index = divisor
                x_arr.append(house_index)
                price_arr.append(y)
                predicted_price_arr.append(y_hat)

        median = np.median(arr)
        mean = np.mean(arr)

        if b_display_graph:
            self.DisplayLineGraph(x_arr, predicted_price_arr, price_arr, graph_title)

        return round(residual / divisor, 2), median, mean

    def FigureOutColumnsToDropBasedOnCorrlation(self, matrix):
        corr_matrix = matrix.corr()
        print(corr_matrix)
        threshold = 0.2
        dict = {}
        # Find pairs with correlation higher than threshold
        high_corr_pairs = np.where(np.abs(corr_matrix) > threshold)
        for i, j in zip(*high_corr_pairs):
            if i < j:  # avoid duplicates and self-correlation
                print(corr_matrix.index[i], corr_matrix.columns[j], corr_matrix.iloc[i, j])
                if corr_matrix.index[i] not in dict:
                    dict[corr_matrix.index[i]] = 1
                else:
                    dict[corr_matrix.index[i]] += 1

                if corr_matrix.columns[j] not in dict:
                    dict[corr_matrix.columns[j]] = 1
                else:
                    dict[corr_matrix.columns[j]] += 1
        print(dict)

    def LeastSquares(self):
        # remember, IRL there is no "perfect" solution - so LSE is able to find the line of best fit with a bias
        # formula = At * Ax = At * b
        A = self.trainingData.copy()
        A['bias'] = 1       # add a column called "bias" and default it to all 1s
        At = A.transpose()

        # b is the "price" matrix - but we want to make sure we only get 80% since the training data is only 80% (will make dataframes the same size)
        b = self.y.loc[self.trainingData.index]
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

    def LeastSquaresPolynomial(self):
        # remember, IRL there is no "perfect" solution - so LSE is able to find the line of best fit with a bias
        # formula = At * Ax = At * b
        A = self.trainingData.copy()
        # pd.set_option('display.max_columns', None)
        # pd.set_option('display.width', 200)
        #
        A['bias'] = 1       # add a column called "bias" and default it to all 1s

        # add squared columns for continuous variables (e.g. area) since other variables are nearly binary
        # find the index of the 'area' column
        # idx = A.columns.get_loc('area')  # returns the integer position of 'area'
        # # insert the new squared column right after it
        # A.insert(loc=idx + 1, column='area_squared', value=(A['area'] ** 2))
        A = A.drop('area', axis=1)

        idx = A.columns.get_loc('furnishingstatus')  # returns the integer position of 'area'
        # insert the new squared column right after it
        A.insert(loc=idx + 1, column='furnishingstatus_squared', value=(A['furnishingstatus'] ** 3))

        # idx = A.columns.get_loc('mainroad')  # returns the integer position of 'area'
        # # insert the new squared column right after it
        # A.insert(loc=idx + 1, column='mainroad_squared', value=(A['mainroad'] ** 1))

        idx = A.columns.get_loc('stories')  # returns the integer position of 'area'
        # insert the new squared column right after it
        A.insert(loc=idx + 1, column='stories_squared', value=(A['stories'] ** 2))

        At = A.transpose()

        # b is the "price" matrix - but we want to make sure we only get 80% since the training data is only 80% (will make dataframes the same size)
        b = self.y.loc[self.trainingData.index]
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

        # add each term to an array and round the float to 8 decimal values
        float_weights = tuple([round(float(val), 8) for val in solution_columns])

        return float_weights