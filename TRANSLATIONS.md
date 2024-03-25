# Translations in Locize

## Before updating docstrings in Locize
Check if any existing key should be updated. If the answer is yes, then make sure it is done according to the rules:
1. If the string has been translated (available in more than one (EN) language), create a new key (only in EN) in the suggested format and assign the new text to it. In
   this case, the current script should be adjusted.
2. If the string is not translated (available only in EN), update the text in the existing key. In this case, the current script should work fine.

For new strings the current script should work fine.

## Updating docstrings in Locize
Run GitHub Actions script to push docstring changes to Locize:
1. Go to the `Actions` tab in the InDSL repository on GitHub
2. Select the `Push JSON to Locize` workflow under `All workflows` on the left hand side
3. Run the workflow in the main branch by clicking the `Run workflow` dropdown on the right hand side and selecting the `Run workflow` option
