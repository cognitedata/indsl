import os

# import docstring_to_json from parse_docstrings.py
import indsl.parse_docstrings as parse_docstrings


def test_json_file_creation():
    # Define a test function with a docstring
    def test_function():
        """This is a test function."""
        pass

    # Create a test module that contains the test function
    test_module = type("test_module", (object,), {"test_function": test_function})

    # Call the docstring_to_json function with the test module
    parse_docstrings.docstring_to_json(test_module)

    # Check whether the expected JSON file has been created
    assert os.path.isfile("toolboxes.json")
