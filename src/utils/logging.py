from typing import List, Union
from logging import LogRecord


def ExclusionFilter(names: Union[str, List[str]]):
    if not isinstance(names, list):
        names = [names]

    def _f(record: LogRecord):
        return 0 if record.name in names else 1

    return _f
