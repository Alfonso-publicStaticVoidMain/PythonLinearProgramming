from ortools.sat.python.cp_model import CpSolverSolutionCallback, IntVar
from typing import Any

from Clases import Trabajador, PuestoTrabajo, Jornada

soluciones: set[frozenset[tuple[Trabajador, PuestoTrabajo, Jornada]]] = set()

class SolutionHandler(CpSolverSolutionCallback):
    def __init__(self, vars: dict[tuple[Trabajador, PuestoTrabajo, Jornada], IntVar]) -> None:
        super().__init__()
        self.vars: dict[tuple[Trabajador, PuestoTrabajo, Jornada], IntVar] = vars
        self._solution_count: int = 0

    def OnSolutionCallback(self) -> None:
        soluciones.add(frozenset({
            (trabajador, puesto, jornada)
            for (trabajador, puesto, jornada) in self.vars
            if self.Value(self.vars[trabajador, puesto, jornada]) == 1
        }))
        self._solution_count += 1

    # Inherited methods from CpSolverSolutionCallback:

    def Value(self, var: IntVar) -> int:
        """Returns the value of a variable in the current solution."""
        return super().Value(var)

    def NumSolutions(self) -> int:
        """Returns the number of solutions found so far."""
        return super().NumSolutions()

    def StopSearch(self) -> None:
        """Instructs the solver to stop search immediately."""
        return super().StopSearch()

    def BooleanValue(self, var: Any) -> bool:
        """Returns the boolean value of a literal (True/False)."""
        return super().BooleanValue(var)

    def WallTime(self) -> float:
        """Returns the elapsed time in seconds since solving started."""
        return super().WallTime()

    def NumConflicts(self) -> int:
        """Returns the number of conflicts encountered so far."""
        return super().NumConflicts()

    def NumBranches(self) -> int:
        """Returns the number of branches explored so far."""
        return super().NumBranches()
