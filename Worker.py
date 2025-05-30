from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Tuple

class ShiftType(Enum):
    MORNING = "morning"
    AFTERNOON = "afternoon"

@dataclass(slots=True)
class Worker:
    id: int
    capabilities: Dict[str, int] # Maps each task to the capability of the worker performing the task.
    #    1 -> Main specialty of the worker, should always be prioritized.
    #    2+ -> Different grades of capability, the higher the number the less capable.
    #    If a task doesn't appear in this dict, it should be assumed the worker is incapable of performing it and should never be assigned it.
    shift_preference: ShiftType # Shift preference for the worker.
    night_volunteer: bool # True if the worker is a volunteer for night shifts. This should be prioritized over its shift preference.
    shift_availability: Dict[str, bool] # Maps each shift to True if the worker is available for it, False otherwise.
    can_double: bool # Represents if the worker is eligible for double shifts or not

    def __init__(self, id: int, capabilities: Dict[str, int], shift_preference: ShiftType, night_volunteer: bool, shift_availability: Dict[str, bool], can_double: bool):
        super().__init__()
        self.can_double = can_double
        self.shift_availability = shift_availability
        self.night_volunteer = night_volunteer
        self.shift_preference = shift_preference
        self.capabilities = capabilities
        self.id = id

    def __post_init__(self):
        self._enforce_types()

    def _enforce_types(self):
        for field_name, field_type in self.__annotations__.items():
            value = getattr(self, field_name)
            if not isinstance(value, field_type):
                raise TypeError(f"Expected {field_name} to be {field_type.__name__}, got {type(value).__name__}")