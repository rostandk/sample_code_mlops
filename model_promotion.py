""" 
CODE SAMPLE by Rostand: https://www.linkedin.com/in/kennezangue/
Code to perform machine learning model deployment through Git instead of making manual changes directely using MLflow GUI.
- The goal is to introduce engineering best practices to ML model deployment: PR, code review, code quality check, model deployment rollback, automatically start A/B testing  etc.
- The data scientists will deploy the ML models through 
  1. changing the ML model configuration file for the environment he/she wants: dev, pre of production
  2. Creating a pull requests
- Above steps will trigger a Github action with all the CI/CD steps: code quality check, unit tests, etc
- After the PR is approved, this ML model is deployed to the choose environment.

Code in relation the MLOPS project from an idea to running A/B experiment in a couple of weeks
- This code is small part of bigger code aiming at adding MLOPS to our entire stack.
- The goal was to signifincanly improve the number of A/B experiments we ran by automating the entire ML pipeline from featuring engineering to model deployment, model monitoring and performing A/B experimenatations.
- This project has allow us to experiment more frequently and deliver quite a lot of business value.

How to run the code:
- This code will trigger automatically by a github action when the data scientist changed the model configuration with the version of the ML model to be roll out or roll back
- CI/CD will run: python promote.py dev
"""

import argparse
import glob
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from time import sleep
from typing import Literal, Optional, Union

import mlflow
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, ConfigDict

FORMAT = "%(asctime)s | %(levelname)s | %(name)s - %(process)d | %(message)s"
logging.basicConfig(format=FORMAT, datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
logger = logging.getLogger(__name__)


# some global constants
BASELINE_ALIAS = "baseline"
CHALLENGER_ALIAS_PREFIX = "challenger"
DEV, PRE, PRO = "dev", "pre", "pro"
CONFIG_PATH = "deploy/models/config"
ML_PROJECT_1, ML_PROJECT_2, ML_PROJECT_3 = "ad_enrichment", "buyers_embeddings", "sellers_embeddings"
MLFLOW_TRACKING_URI = "http://127.0.0.1:8001/api/v1/namespaces/ml/services/mlflow:http/proxy"
# MLFLOW_TRACKING_URI = "http://127.0.0.1:8080"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


@dataclass
class Result:
    """Result class to host success or error message from a call to MLFlow model registry.
    Args:
        - `message (str)`: success or error message that we will display to the user.
        - `error (boolean)`: value is True if call return the intended ressource. False otherwise.
    """

    message: str
    error: bool


class Configuration(BaseModel):
    """Pydantic Basemodel class that validate and load json config file with model to promote.
    Args:
        - `model_version (str)`: version number of the model that will be promoted to baseline
        - `model_env (literal)`: MLFlow environment where we wish to promote the model to baseline
        - `model_name (literal)`: MLFlow registered model name. Possible values are: ML_PROJECT_1, ML_PROJECT_2, ML_PROJECT_3
        - `model_description (optional str)`: description (algo name) of the model we are promoting
        - `model_alias (literal)`: The alias that will attached to the baseline model: `baseline`
        -  TODO: implement section that actually add description to model version.
    """

    model_version: str
    model_env: Literal[DEV, PRE, PRO]
    model_name: Literal[ML_PROJECT_1, ML_PROJECT_2, ML_PROJECT_3]
    model_description: Optional[str] = None
    model_alias: Literal[BASELINE_ALIAS] = BASELINE_ALIAS
    # Please check: adrs/dataeng/adr_automated_retraining.md

    # to silence warnings about model_ being a reserved Pydantic variable name
    model_config = ConfigDict(protected_namespaces=())

    @classmethod
    def load_model_config(cls, path_config_file: Union[str, Path]) -> "Configuration":
        """Load json config file, perform data validation and return Configuration class
        Args:
            path_config_file (Union[str, Path]): path_config_file: full path to json config file
        Returns:
            Configuration: configuration as a Pydantic basemodel class
            Raise Pydantic Exception in case of validation errors
        """
        with open(path_config_file, mode="r") as f:
            return cls(**json.load(f))


class BaselinePromoter:
    """Promote ML model by:
    - Adding the alias baseline
    - Removing all challenger aliases for that model version.
    An example of challenger aliases that will be removed:
    - challenger_ar
    - challenger_experiment_name
    Args:
        client (MlflowClient): Mlflow client
        env : MLFlow environment where the promotion happens
        config (Configuration): Model configuration loaded from json config file
    """

    def __init__(self, client: MlflowClient, config: Configuration, env: Literal[DEV, PRE, PRO]):
        self.env = env
        self.client: MlflowClient = client
        self.model_version: str = config.model_version
        self.model_alias: Literal[BASELINE_ALIAS] = config.model_alias
        self.model_name: Literal[ML_PROJECT_1, ML_PROJECT_2, ML_PROJECT_3] = config.model_name

    def _get_model_by_alias(self, model_name: str, model_alias: str) -> Optional[ModelVersion]:
        """Get model by a name and an alias from the MLFlow model registry.
        Args:
            model_name: registered model name
            model_alias: model alias example are: baseline, challenger_ar
        Returns:
            MLflow ModelVersion
        """
        try:
            model: ModelVersion = self.client.get_model_version_by_alias(name=model_name, alias=model_alias)
            logger.debug(model)
            return model
        except MlflowException as mlflow_exception:
            logger.debug(
                f"model name:{model_alias}; model_alias:{model_alias}; "
                f"Exception: {mlflow_exception.serialize_as_json()}"
            )
            return

    def _add_alias_to_model(self, model_name: str, model_version: str, model_alias: str) -> Result:
        """Promote ML model by adding the alias `baseline` to a specific model version.
        Note that this will remove any existing baseline alias.
        Args:
            - `model_version (str)`: version number of the model that will be promoted to baseline
            - `model_alias (literal)`: The alias that will attached to the baseline model: `baseline`
            - `model_name (literal)`: MLFlow registered model name. Possible values are:ML_PROJECT_1, ML_PROJECT_2, ML_PROJECT_3
        Returns:
            - Result class
        """
        sucess_msg = (
            f"We have SUCESSFULLY added alias: `{model_alias}` to "
            f"version: `{model_version}` of the model name: `{model_name}`\n"
        )
        failure_msg = (
            f"We FAILED to add alias: `{model_alias}` to version: `{model_version}` of the model name: `{model_name}`\n"
        )

        try:
            # Add baseline alias to model name and version in the model registry
            # Remove an existing baseline alias
            self.client.set_registered_model_alias(name=model_name, alias=model_alias, version=model_version)
            return Result(error=False, message=sucess_msg)
        except MlflowException as mlflow_exception:
            failure_msg = f"{failure_msg} Exception: {mlflow_exception.serialize_as_json()}"
            logger.debug(failure_msg)
            return Result(error=True, message=failure_msg)

    def _remove_alias_from_model(self, model_name: str, model_alias: str) -> Result:
        """Delete an alias associated with a registered model.
        Args:
            `model_alias (literal)`: The alias that will attached to the baseline model: `baseline`
            `model_name (literal)`: MLFlow registered model name. Possible values are:ML_PROJECT_1, ML_PROJECT_2, ML_PROJECT_3
        Returns:
                Result class
        """
        sucess_msg = f"We have SUCESSFULLY deleted model alias: `{model_alias}` from model name: `{model_name}`\n"
        failure_msg = f"We have FAILED to remove model alias: `{model_alias}` from model name: `{model_name}`."

        try:
            self.client.delete_registered_model_alias(model_name, model_alias)
            return Result(error=False, message=sucess_msg)
        except MlflowException as mlflow_exception:
            msg = f"{failure_msg} Exception: {mlflow_exception.serialize_as_json()}"
            logger.debug(msg)
            return Result(error=True, message=msg)

    def _remove_challenger_aliases_from_model(
        self, model_name: str, model_version: str, prefix: str = CHALLENGER_ALIAS_PREFIX
    ) -> None:
        """Remove all challenger aliases from the model that will be promoted.
        The assumption is that a model that will be promoted to baseline have sucessfully concluded all A/B tests.
        It should therefore loose all his challenger aliases.

        TODO:
        - We should decide if we throw an exception when failing to remove challenger alias from the model being
          promoted.
        - At the moment, we just providing a warning in the form of logger.error with instruction.

        Args:
            `model_name (literal)`: MLFlow registered model name. Possible values are: ML_PROJECT_1, ML_PROJECT_2, ML_PROJECT_3
            `model_version (str)`: version number of the model that will be promoted to baseline
            `prefix (str)`: Prefix challenger aliases that will be remove from model that will be promoted.
        """
        # get the model that will be promoted by name and version
        found_model: ModelVersion = self.client.get_model_version(name=model_name, version=model_version)
        if found_model and found_model.aliases:
            for alias in found_model.aliases:
                if prefix in alias:
                    logger.info(f"Removing alias:`{alias}` from model: `{model_name}` version:`{model_version}`.")
                    # remove alias from registered model: model_name
                    result = self._remove_alias_from_model(model_name=model_name, model_alias=alias)
                    if not result.error:
                        logger.info(f"Yes, `{alias}` was removed.")
                    elif result.error:
                        logger.error(
                            f"Warning: we failed to remove alias: {alias} from "
                            f"model:`{model_name}` version:`{model_version}`."
                        )

    def _verify_that_baseline_matches_version(self, model_name: str, model_alias: str, model_version: str) -> Result:
        """Verification step: Does the baseline model version matches version in configuration file ?
        Args:
        - `model_alias (literal)`: The alias that will attached to the baseline model: `baseline`
        - `model_name (literal)`: MLFlow registered model name. Possible values are: ML_PROJECT_1, ML_PROJECT_2, ML_PROJECT_3
        - `model_version (str)`: version number of the model that will be promoted to baseline
        Returns:
            Result class
        TODO: We should decide if we throw an exceptio when if baseline model still have challenger alias.
        """
        found_model: Optional[ModelVersion] = self._get_model_by_alias(model_name=model_name, model_alias=model_alias)
        if not found_model:
            failure_msg = f"Weird, no baseline model was not found on the registry for model name: {model_name}"
            logger.debug(failure_msg)
            return Result(message=failure_msg, error=True)
        # check if new baseline model version has no challenger aliases
        # turn list of aliases into string and check for prefix string `challenger`
        challenger_aliases = [alias for alias in found_model.aliases if CHALLENGER_ALIAS_PREFIX in alias]
        has_challenger_aliases = True if len(challenger_aliases) > 0 else False

        message = f"model name: `{model_name}`; model version: `{model_version}`; model alias: `{model_alias}`"
        # check if baseline model version has the correct version and no challenger aliases
        if found_model and found_model.version == model_version and not has_challenger_aliases:
            sucess_msg = f"We have SUCESSFULLY verified that the baseline model has the correct version. {message}"
            logger.debug(sucess_msg)
            return Result(message=sucess_msg, error=False)

        elif found_model and found_model.version == model_version and has_challenger_aliases:
            failure_msg = (
                f"WARNING: Baseline model still have some challengers aliases: {challenger_aliases}. Check: {message}."
            )
            logger.error(failure_msg)
            return Result(message=failure_msg, error=True)
        else:
            failure_msg = (
                f"We FAILED to verify that the baseline model has the correct model version. Please check: {message}.\n"
            )
            logger.error(failure_msg)
            return Result(message=failure_msg, error=True)

    def start_model_promotion(self):
        """entry function that start model promotion mentioned in the config file"""

        # variable to keep the current baseline model, to provide instruction on how to rollback in case of error
        current_baseline_version: Optional[ModelVersion] = self._get_model_by_alias(
            model_name=self.model_name, model_alias=BASELINE_ALIAS
        )
        # variable to keep the current challenger model, to provide instruction on how to rollback in case of error
        current_challenger_version: Optional[ModelVersion] = self.client.get_model_version(
            name=self.model_name, version=self.model_version
        )
        # 0. check if the combination of model name, model alias and model version exist already on the registry
        # if True, nothing more to do, the correct model version is already the baseline else proceed with promotion
        found_model: Optional[ModelVersion] = self._get_model_by_alias(
            model_name=self.model_name, model_alias=self.model_alias
        )
        if found_model and found_model.version == self.model_version:
            logger.info(
                f"Model alias: `{self.model_alias}` for model: `{self.model_name}`and "
                f"version:`{self.model_version}` already exist. No changes needed.\n"
            )
            return

        # 1. Remove existing challenger aliases from the model version that will be promoted
        # if we are promoting a model to baseline, it means that it loose the challenger status
        # we will remove all chalenger aliases including: challenger_ar, challenger_experiment_name
        self._remove_challenger_aliases_from_model(model_name=self.model_name, model_version=self.model_version)

        # 2. Promote model version (in var model_version) to baseline
        # 2.a add baseline alias to that specific version
        logger.info(
            f"Promoting model: {self.model_name} with version: {self.model_version} "
            f"to: `{self.model_alias}` on `{self.env}` environment."
        )
        adding: Result = self._add_alias_to_model(
            model_name=self.model_name, model_version=self.model_version, model_alias=self.model_alias
        )
        if adding.error:
            logger.error(
                f"We failed to add alias: `{self.model_alias}` to model: "
                f"{self.model_name} - {self.model_version} on `{self.env}` environment.\n"
            )
        else:
            logger.info(
                f"Hoera, promoting model: {self.model_name} - {self.model_version} "
                f"to: `{self.model_alias}` was succesfull."
            )

        # 2.b verify if the baseline alias was really added to the correct model version
        # There is a small delay between adding an alias and checking if that alias actually exist.
        # need to sleep and account for that delay.
        sleep(5)
        verification: Result = self._verify_that_baseline_matches_version(
            model_name=self.model_name, model_alias=self.model_alias, model_version=self.model_version
        )
        if verification.error:
            logger.error(f"{verification.message}.\n")
            logger.error(
                "ROLLBACK INSTRUCTION:\nGo to the MLFlow model registry to manually rollback:\n"
                f" - Make sure the baseline model has the following alias and version: {current_baseline_version}\n"
                f" - Make sure the challenger model has the following alias and version: {current_challenger_version}\n"
                f" - Manual rerun this Github Actions pipeline to restart the promotion process.\n"
            )
            raise RuntimeError(verification.message)
        else:
            logger.info(
                f"Awesome, we have verified that model: {self.model_name} - {self.model_version} "
                f"has the alias: `{self.model_alias}`."
            )
            return


def get_environment_variable_from_input_args() -> Literal[DEV, PRE, PRO]:
    """Get environment variable after script is called.
    Example: promote.py env where env = dev or pre or pro.
    The extracted environemnt var will allow us to load the proper configuration file.
    Returns:
        one of the following values: "dev", "pre", "pro"
        Raise ArgumentError exception when the user provides a value different from above
    """
    parser = argparse.ArgumentParser()
    arg_description = "You must specify an environment. Allowed values are: dev, pre, pro. Example: promote.py pro"
    parser.add_argument("env", help=arg_description, choices=[DEV, PRE, PRO])
    args = parser.parse_args()
    return args.env


def main():
    # extract ennvironment variable in order to load the proper json configuration
    env = get_environment_variable_from_input_args()
    mlflow_client = MlflowClient()
    path_pattern = f"{CONFIG_PATH}/{env}/*.json"
    logger.debug(f"path pattern:  {path_pattern}")
    for path_config_file in glob.iglob(path_pattern):
        logger.info(f"Start proccessing content of config file: {path_config_file}")
        config: Configuration = Configuration.load_model_config(path_config_file)
        promoter = BaselinePromoter(client=mlflow_client, config=config, env=env)
        promoter.start_model_promotion()


if __name__ == "__main__":
    main()

