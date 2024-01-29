import os

import translations.create_translation_json as create_translation


def test_json_file_creation():
    # Define a test function with a docstring with parameters, return value and description
    def test_function(test_parameter: str, test_return: str = "test_return") -> str:
        """Test function.

        This is a long description of the test function. It can contain multiple lines. It can also contain
        links (https://www.cognite.com).

        Args:
            test_parameter: Test parameter.
                Test parameter description.
            test_return: Test return.
                Test return description.

        Returns:
            pd.Series: Test return.
        """
        return test_return

    # Create a test module that contains the test function
    test_module = type("test_module", (object,), {"test_function": test_function("test_parameter", "test_return")})

    create_translation.create_mapping_for_translations(test_module)

    # Check whether the expected JSON file has been created
    assert os.path.isfile("translated_toolboxes.json")
