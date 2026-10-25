from src.config import load_config


def test_config_contains_expected_dataset_schema():
    config = load_config()
    assert config["data"]["target_column"] == "Class"
    assert len(config["data"]["expected_columns"]) == 31
    assert config["data"]["expected_columns"][0] == "Time"
    assert config["data"]["expected_columns"][-1] == "Class"
