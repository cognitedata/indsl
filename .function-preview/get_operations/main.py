import json

from data_science_service import data_science_service
from schemas import OperationSchema


def get_operations(include_deprecated: bool = False):
    operations = data_science_service.get_available_operations()
    schema = OperationSchema(context={"include_deprecated": include_deprecated})
    return schema.dump(operations, many=True)


operations = get_operations()
jsonString = json.dumps(operations, indent=2)

with open(".function-preview/src/assets/response.json", "w") as jsonFile:
    jsonFile.write(jsonString)
