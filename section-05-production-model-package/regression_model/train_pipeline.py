import numpy as np
from config.core import config
# Import the pipeline "price_pipe" from the other module
from pipeline import price_pipe
# Import the functions to import dataset and export pipeline from the data_manager module
from processing.data_manager import load_dataset, save_pipeline
from sklearn.model_selection import train_test_split


def run_training() -> None:
    """Train the model."""

    # read training data - import from the file address stated in the config file.
    data = load_dataset(file_name=config.app_config.training_data_file)

    # divide train and test - all the details (features/targets/test size/random states) are includedin the config file.
    X_train, X_test, y_train, y_test = train_test_split(
        data[config.model_config.features],  # predictors
        data[config.model_config.target],
        test_size=config.model_config.test_size,
        # we are setting the random seed here
        # for reproducibility
        random_state=config.model_config.random_state,
    )
    y_train = np.log(y_train)

    # fit model
    price_pipe.fit(X_train, y_train)

    # persist trained model
    save_pipeline(pipeline_to_persist=price_pipe)


if __name__ == "__main__":
    run_training()
