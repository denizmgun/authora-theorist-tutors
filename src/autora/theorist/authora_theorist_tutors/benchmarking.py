from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# autora state
from autora.state import StandardState, on_state, Delta

# experiment_runner
from autora.experiment_runner.synthetic.psychophysics.weber_fechner_law import weber_fechner_law
from autora.experiment_runner.synthetic.psychophysics.stevens_power_law import stevens_power_law
from autora.experiment_runner.synthetic.economics.expected_value_theory import expected_value_theory

# experimentalist
from autora.experimentalist.grid import grid_pool
from autora.experimentalist.random import random_pool, random_sample

# bayesian theorist
import autora.theorist.bms
from autora.theorist.bms.regressor import BMSRegressor


# data handling
from sklearn.model_selection import train_test_split


def benchmark(experiment_runner, theorist):

    # Give the dummy and bms theorists
    dummy_theorist = DummyTheorist()
    #bms_theorist = BMSRegressor() #autora.theorist.bms()

    # generate all conditions
    conditions = experiment_runner.domain()

    # generate all corresponding observations
    experiment_data = experiment_runner.run(conditions, added_noise=0.01)

    # get the name of the independent and independent variables
    ivs = [iv.name for iv in experiment_runner.variables.independent_variables]
    dvs = [dv.name for dv in experiment_runner.variables.dependent_variables]

    # extract the dependent variable (observations) from experiment data
    conditions = experiment_data[ivs]
    observations = experiment_data[dvs]

    # split into train and test datasets
    conditions_train, conditions_test, observations_train, observations_test = train_test_split(conditions, observations)

    print("#### EXPERIMENT CONDITIONS (X):")
    print(conditions)
    print("#### EXPERIMENT OBSERVATIONS (Y):")
    print(observations)

    # fit theorist
    theorist.fit(conditions_train, observations_train)

    # compute prediction for validation set
    predictions_theorist = theorist.predict(conditions_test)

    # fit dummy theorist
    dummy_theorist.fit(conditions_train, observations_train)

    # compute prediction for dummy theorist
    predictions_dummy_theorist = dummy_theorist.predict(conditions_test)

    # fit BMS theorist
    #bms_theorist.fit(conditions_train, observations_train)

    # compute prediction for BMS theorist
    #predictions_bms_theorist = bms_theorist.predict(conditions_test)

    # evaluate theorist performance
    error_theorist = (predictions_theorist - observations_test).pow(2)
    error_theorist = error_theorist.mean()

    # evaluate dummy theorist performance
    error_dummy_theorist = (predictions_dummy_theorist - observations_test).pow(2)
    error_dummy_theorist = error_dummy_theorist.mean()

    # evaluate BMS theorist performance
    #error_bms_theorist = (predictions_bms_theorist - observations_test).pow(2)
    #error_bms_theorist = error_bms_theorist.mean()

    print("#### IDENTIFIED EQUATION OF THE THEORIST:")
    print(theorist.print_eqn())

    print("#### IDENTIFIED EQUATION OF THE DUMMY THEORIST:")
    print(dummy_theorist.print_eqn())

    print("#### IDENTIFIED EQUATION OF THE BMS THEORIST:")
    #bms_theorist.present_results()

    print("#### ERROR - THEORIST:")
    print(error_theorist)

    print("#### ERROR - DUMMY THEORIST:")
    print(error_dummy_theorist)

    print("#### ERROR - BMS THEORIST:")
    #print(error_bms_theorist)





class DummyTheorist:
    """
    This theorist fits a polynomial function to the data.
    """

    def __init__(self, degree: int = 3):
      self.poly = PolynomialFeatures(degree=degree, include_bias=False)
      self.model = LinearRegression()

    def fit(self, x, y):
      features = self.poly.fit_transform(x, y)
      self.model.fit(features, y)
      return self

    def predict(self, x):
      features = self.poly.fit_transform(x)
      return self.model.predict(features)

    def print_eqn(self):
        # Extract the coefficients and intercept
        coeffs = self.model.coef_
        intercept = self.model.intercept_

        # Handle multi-output case by iterating over each output's coefficients and intercept
        if coeffs.ndim > 1:
            for idx in range(coeffs.shape[0]):
                equation = f"y{idx+1} = {intercept[idx]:.3f}"
                feature_names = self.poly.get_feature_names_out()
                for coef, feature in zip(coeffs[idx], feature_names):
                    equation += f" + ({coef:.3f}) * {feature}"
                print(equation)
        else:
            equation = f"y = {intercept:.3f}"
            feature_names = self.poly.get_feature_names_out()
            for coef, feature in zip(coeffs, feature_names):
                equation += f" + ({coef:.3f}) * {feature}"
            print(equation)

#dummy_theorist = DummyTheorist()

def run_benchmark(theorist):
   
   experiment_runners = [stevens_power_law(), weber_fechner_law(), expected_value_theory()]

   for runner in experiment_runners:
      
      benchmark(runner, theorist)
   


theorist = DummyTheorist()
benchmark(stevens_power_law(), theorist)