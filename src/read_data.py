from ucimlrepo import fetch_ucirepo


def load_diabetes_data(verbos=True):
    # fetch dataset
    diabetes_130_us_hospitals_for_years_1999_2008 = fetch_ucirepo(id=296)

    # data (as pandas dataframes)
    X = diabetes_130_us_hospitals_for_years_1999_2008.data.features
    y = diabetes_130_us_hospitals_for_years_1999_2008.data.targets

    if verbos:
        # metadata
        print(diabetes_130_us_hospitals_for_years_1999_2008.metadata)

        # variable information
        print(diabetes_130_us_hospitals_for_years_1999_2008.variables)
