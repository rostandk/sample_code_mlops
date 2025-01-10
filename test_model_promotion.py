"""
Unit tests for model_promotion.py
pytest test_model_promotion.py
"""
import pytest
import json
from unittest.mock import MagicMock
from pathlib import Path
from pydantic import ValidationError
from mlflow.exceptions import MlflowException
from mlflow.entities.model_registry import ModelVersion
from model_promotion import Configuration, BaselinePromoter

MODEL_NAME = "finetuned_llama_3_2_for_ad_enrichment"

class ModelPromotionTest:
    @pytest.fixture
    def valid_config(self):
        return {
            "model_version": "1",
            "model_env": "dev",
            "model_name": MODEL_NAME,
            "model_description": "Test model",
            "model_alias": "baseline"
        }

    @pytest.fixture
    def invalid_config(self):
        return {
            "model_version": "1",
            "model_env": "invalid_env",
            "model_name": MODEL_NAME,
            "model_alias": "baseline"
        }

    @pytest.fixture
    def mock_client(self):
        return MagicMock()

    def test_load_model_config_valid(self, valid_config):
        path = "test_config_valid.json"
        with open(path, "w") as f:
            json.dump(valid_config, f)

        config = Configuration.load_model_config(path)
        assert config.model_version == "1"
        assert config.model_env == "dev"
        assert config.model_name == MODEL_NAME
        assert config.model_alias == "baseline"

    def test_load_model_config_invalid(self, invalid_config):
        path = "test_config_invalid.json"
        with open(path, "w") as f:
            json.dump(invalid_config, f)

        with pytest.raises(ValidationError):
            Configuration.load_model_config(path)

    def test_baseline_promoter_initialization(self, mock_client, valid_config):
        config = Configuration(**valid_config)
        promoter = BaselinePromoter(client=mock_client, config=config, env="dev")

        assert promoter.env == "dev"
        assert promoter.client == mock_client
        assert promoter.model_version == "1"
        assert promoter.model_alias == "baseline"
        assert promoter.model_name == MODEL_NAME

    def test_get_model_by_alias_success(self, mock_client):
        mock_client.get_model_version_by_alias.return_value = ModelVersion("name", "1", "1")
        promoter = BaselinePromoter(client=mock_client, config=None, env="dev")

        result = promoter._get_model_by_alias(MODEL_NAME, "baseline")
        assert result is not None
        mock_client.get_model_version_by_alias.assert_called_with(name=MODEL_NAME, alias="baseline")

    def test_get_model_by_alias_failure(self, mock_client):
        mock_client.get_model_version_by_alias.side_effect = MlflowException("Error")
        promoter = BaselinePromoter(client=mock_client, config=None, env="dev")

        result = promoter._get_model_by_alias(MODEL_NAME, "baseline")
        assert result is None

    def test_add_alias_to_model_success(self, mock_client):
        promoter = BaselinePromoter(client=mock_client, config=None, env="dev")

        result = promoter._add_alias_to_model(MODEL_NAME, "1", "baseline")
        assert not result.error
        mock_client.set_registered_model_alias.assert_called_with(
            name=MODEL_NAME, alias="baseline", version="1"
        )

    def test_add_alias_to_model_failure(self, mock_client):
        mock_client.set_registered_model_alias.side_effect = MlflowException("Error")
        promoter = BaselinePromoter(client=mock_client, config=None, env="dev")

        result = promoter._add_alias_to_model(MODEL_NAME, "1", "baseline")
        assert result.error

    def test_remove_alias_from_model_success(self, mock_client):
        promoter = BaselinePromoter(client=mock_client, config=None, env="dev")

        result = promoter._remove_alias_from_model(MODEL_NAME, "baseline")
        assert not result.error
        mock_client.delete_registered_model_alias.assert_called_with(MODEL_NAME, "baseline")

    def test_remove_alias_from_model_failure(self, mock_client):
        mock_client.delete_registered_model_alias.side_effect = MlflowException("Error")
        promoter = BaselinePromoter(client=mock_client, config=None, env="dev")

        result = promoter._remove_alias_from_model(MODEL_NAME, "baseline")
        assert result.error

    def test_verify_that_baseline_matches_version_success(self, mock_client):
        mock_model_version = ModelVersion("name", "1", "1")
        mock_client.get_model_version_by_alias.return_value = mock_model_version

        promoter = BaselinePromoter(client=mock_client, config=None, env="dev")
        result = promoter._verify_that_baseline_matches_version(MODEL_NAME, "baseline", "1")

        assert not result.error

    def test_verify_that_baseline_matches_version_failure(self, mock_client):
        mock_model_version = ModelVersion("name", "1", "2")
        mock_client.get_model_version_by_alias.return_value = mock_model_version

        promoter = BaselinePromoter(client=mock_client, config=None, env="dev")
        result = promoter._verify_that_baseline_matches_version(MODEL_NAME, "baseline", "1")

        assert result.error
