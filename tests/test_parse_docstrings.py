import os
import json
import indsl
import pathlib

import translations.create_translation_json as create_translation


def test_json_file_creation():
    # Create a JSON file with the translation keys and values for the full indsl package
    output_dictionary = create_translation.create_mapping_for_translations(indsl)

    # Check whether the expected JSON file has been created
    assert os.path.isfile("translated_toolboxes.json")

    # check that they keys and values are output_dictionary
    assert output_dictionary.get("INDSL_DRIFT_1.0") == "Drift"
    assert output_dictionary.get("INDSL_DRIFT_DESCRIPTION_1.0") == (
        "This function detects data drift (deviation) by comparing two rolling averages, short and long interval, of the signal. The\ndeviation between the short and long term average is considered significant if it is above a given threshold\nmultiplied by the rolling standard deviation of the long term average."
    )
    assert output_dictionary.get("INDSL_DRIFT_DATA_1.0") == "Time series"
    assert output_dictionary.get("INDSL_DRIFT_LONG_INTERVAL_1.0") == "Long length"
    assert output_dictionary.get("INDSL_DRIFT_LONG_INTERVAL_DESCRIPTION_1.0") == "Length of long term time interval."
    assert output_dictionary.get("INDSL_DRIFT_SHORT_INTERVAL_1.0") == "Short length"
    assert output_dictionary.get("INDSL_DRIFT_SHORT_INTERVAL_DESCRIPTION_1.0") == "Length of short term time interval."
    assert output_dictionary.get("INDSL_DRIFT_STD_THRESHOLD_1.0") == "Threshold"
    assert output_dictionary.get("INDSL_DRIFT_STD_THRESHOLD_DESCRIPTION_1.0") == (
        "Parameter that determines if the signal has changed significantly enough to be considered drift. The threshold\nis multiplied by the long term rolling standard deviation to take into account the recent condition of the\nsignal."
    )
    assert output_dictionary.get("INDSL_DRIFT_DETECT_1.0") == "Type"
    assert output_dictionary.get("INDSL_DRIFT_DETECT_DESCRIPTION_1.0") == (
        'Parameter to determine if the model should detect significant decreases, increases or both. Options are:\n"decrease", "increase", or "both". Defaults to "both".'
    )
    assert output_dictionary.get("INDSL_DRIFT_RETURN_1.0") == "Boolean time series"
