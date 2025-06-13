import random
from pickle import FALSE

from ortools.sat.python import cp_model
from ortools.sat.python.cp_model import CpModel, IntVar
from typing import List, Optional, Dict, Tuple


def solve_assignment(
        workers: List[int | str],  # Contains the IDs or names of the workers
        tasks: List[int | str],  # Contains the different tasks, identified by an int or a string
        shifts: List[str], # Contains the names of the different shifts. For now, they are assumed ['morning', 'afternoon', 'night_1', 'night_2']
        demand: Dict[Tuple[int | str, str], int], # For each task and shift, maps it to the number of workers demanded for it
        specialties: Dict[int | str, List[int | str]], # For each task, maps it to a List of the workers that have that task as their specialty, ordered by their priority of assignment
        capabilities: Dict[int | str, List[int | str]],
        # For each task, maps it to a List of the workers that can perform that task without being their specialty
        worker_availability: Dict[Tuple[int | str, str], bool], # For each worker and shift, maps it to true or false depending on if the worker is available for that shift
        verbose: bool = False,  # Parameter to print the result via console or not
        max_score: int = 1000,  # Base score a single assignment can reach
        specialty_weight: float = 1.0,  # Weight for assigning a worker to its specialty
        capability_weight: float = 0.3,  # Weight for assigning a worker to a task that isn't their specialty
        priority_decay: int = 10,  # Base score loss for being further from the top of the list
) -> Dict[Tuple[int | str, int | str, str], bool]:
    model: CpModel = cp_model.CpModel()

    # Define the variables on a dictionary
    x: Dict[Tuple[int | str, int | str, str], IntVar] = {}
    for w in workers:
        for t in tasks:
            for s in shifts:
                # A worker cannot be assigned to a task they can't perform or is not available for
                # If that's the case, do not even create the variable
                if w in specialties[t] + capabilities[t] and worker_availability.get((w, s), False):
                    x[w, t, s] = model.NewBoolVar(f'x_{w}_{t}_{s}')

    for w in workers:
        # Each worker can have at most 2 shifts
        model.Add(sum(x.get((w, t, s), 0) for t in tasks for s in shifts) <= 2)
        for s in shifts:
            # Each worker can work at most 1 task each shift
            model.Add(sum(x.get((w, t, s), 0) for t in tasks) <= 1)

        # A worker that works double shifts mustn't do it during night
        is_double_shift = model.NewBoolVar(f'double_shift_{w}')
        total_shifts = sum(x.get((w, t, s), 0) for t in tasks for s in shifts)
        model.Add(total_shifts == 2).OnlyEnforceIf(is_double_shift)
        model.Add(total_shifts != 2).OnlyEnforceIf(is_double_shift.Not())

        night_shifts = sum(x.get((w, t, s), 0) for t in tasks for s in ['night_1', 'night_2'])
        model.Add(night_shifts == 0).OnlyEnforceIf(is_double_shift)

    # Each task must meet its demand exactly on each shift
    for t in tasks:
        for s in shifts:
            model.Add(sum(x.get((w, t, s), 0) for w in workers) == demand[t, s])

    # Define the score of assigning a worker to a task on a shift, according to its position on the lists and if it's their specialty
    scores: Dict[Tuple[int | str, int | str, str], float] = {}
    for t in tasks:
        for s in shifts:
            for rank, w in enumerate(specialties[t]):
                scores[w, t, s] = int(specialty_weight * (max_score - priority_decay * rank))
            for rank, w in enumerate(capabilities[t]):
                if (w, t, s) not in scores:
                    scores[w, t, s] = int(capability_weight * (max_score - priority_decay * rank))
            for w in workers:
                scores.setdefault((w, t, s), 0)

    # Create a List of variables where it represents the assignment of a worker into its specialty
    specialty_assigned: List[IntVar] = []
    for (w, t, s), var in x.items():
        if w in specialties[t]:
            specialty_assigned.append(var)

    # Set the first step of the model to maximize specialty assignments
    model.Maximize(sum(specialty_assigned))

    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 8
    solver.parameters.search_branching = cp_model.FIXED_SEARCH
    status = solver.Solve(model)

    if verbose:
        print("Advanced usage: Step 1")
        print(f"Problem solved in {format_duration(solver.wall_time)}")
        print(f"Conflicts: {solver.NumConflicts()}")
        print(f"Branches: {solver.NumBranches()}\n")

    if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        if verbose: print("No solution found in step 1.")
        return {}

    # Add a constraint to the model to reach the maximum specialty assignments
    model.Add(sum(specialty_assigned) == int(solver.ObjectiveValue()))

    # Set the new function to maximize, a sum of the assignments pondered by their scores
    model.Maximize(sum(scores[w, t, s] * x[w, t, s] for (w, t, s) in x))

    # Solve the model with the given restrictions and objective
    solver2 = cp_model.CpSolver()
    solver2.parameters.num_search_workers = 8
    solver2.parameters.search_branching = cp_model.FIXED_SEARCH
    status = solver2.solve(model)

    if verbose:
        print("Advanced usage: Step 2")
        print(f"Problem solved in {format_duration(solver.wall_time)}")
        print(f"Conflicts: {solver.NumConflicts()}")
        print(f"Branches: {solver.NumBranches()}\n")

    result: Dict[Tuple[int | str, int | str, str], bool] = {}
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        for w in workers:
            for t in tasks:
                for s in shifts:
                    result[w, t, s] = bool(solver.Value(x[w, t, s])) if (w, t, s) in x else False

        if verbose:
            if status == cp_model.OPTIMAL:
                print('Optimal solution found.')
            else:
                print('Feasible solution found. Optimality cannot be guaranteed.')

            print(
                f'Number of specialty assignments achieved = {solver.ObjectiveValue()} out of {sum(demand.get((t, s), 0) for t in tasks for s in shifts)}\n')
            for w in workers:
                for t in tasks:
                    for s in shifts:
                        if result[w, t, s]:
                            print(f'Worker {w} assigned to Task {t} on the {s} shift')
            print("")
            for t in tasks:
                for s in shifts:
                    print(f'Task {t} on {s} shift assigned to:')
                    for w in workers:
                        if result[w, t, s]:
                            print(f'\tWorker {w}')
    else:
        if verbose: print('No feasible solution found.')

    return result


def format_duration(seconds: float | int) -> str:
    if seconds < 10:
        return f"{seconds} seconds"
    seconds_int = int(seconds)
    if seconds_int < 60:
        return f"{seconds_int} seconds"
    elif seconds_int < 3600:
        minutes = seconds_int // 60
        rem_seconds = seconds_int % 60
        return f"{minutes} minutes and {rem_seconds} seconds"
    else:
        hours = seconds_int // 3600
        rem_seconds = seconds_int % 3600
        minutes = rem_seconds // 60
        rem_seconds = rem_seconds % 60
        return f"{hours} hours, {minutes} minutes and {rem_seconds} seconds"


def printTable(
        solution: Dict[Tuple[int | str, int | str, str], bool],
        workers: List[int | str],
        tasks: List[int | str],
        shifts: List[str],
        col_width: int = 4,
        v_sep: str = "│",
        h_line: str = "─",
        cross: str = "┼",
) -> None:
    for s in shifts:
        print(f"{s.capitalize()} shift")

        # Header row
        header = " " * col_width + v_sep + v_sep.join(f"{w:^{col_width}}" for w in workers) + v_sep
        print(header)

        # Separator row with ┼ intersections
        sep = h_line * col_width + cross
        for _ in range(len(workers) - 1):
            sep += h_line * col_width + cross
        sep += h_line * col_width
        print(sep)

        # Data rows
        for t in tasks:
            row = f"{str(t):^{col_width}}" + v_sep
            for w in workers:
                val = solution.get((w, t, s), False)
                row += f"{'X' if val else ' ':^{col_width}}" + v_sep
            print(row)
            print(sep)

        print()


workers: List[int | str] = list(range(50))
tasks: List[int | str] = list(range(10))
shifts: List[str] = ['morning', 'afternoon', 'night_1', 'night_2']
demand: Dict[Tuple[int | str, str], int] = {
    (0, 'morning'): 2, (0, 'afternoon'): 2, (0, 'night_1'): 1, (0, 'night_2'): 1,
    (1, 'morning'): 1, (1, 'afternoon'): 2, (1, 'night_1'): 2, (1, 'night_2'): 2,
    (2, 'morning'): 1, (2, 'afternoon'): 1, (2, 'night_1'): 2, (2, 'night_2'): 1,
    (3, 'morning'): 2, (3, 'afternoon'): 1, (3, 'night_1'): 1, (3, 'night_2'): 2,
    (4, 'morning'): 1, (4, 'afternoon'): 1, (4, 'night_1'): 1, (4, 'night_2'): 2,
    (5, 'morning'): 2, (5, 'afternoon'): 1, (5, 'night_1'): 2, (5, 'night_2'): 2,
    (6, 'morning'): 2, (6, 'afternoon'): 1, (6, 'night_1'): 1, (6, 'night_2'): 1,
    (7, 'morning'): 1, (7, 'afternoon'): 1, (7, 'night_1'): 2, (7, 'night_2'): 2,
    (8, 'morning'): 1, (8, 'afternoon'): 2, (8, 'night_1'): 1, (8, 'night_2'): 1,
    (9, 'morning'): 2, (9, 'afternoon'): 2, (9, 'night_1'): 2, (9, 'night_2'): 1,
}
specialties: Dict[int | str, List[int | str]] = {
    0: [0, 1, 2, 3, 4],
    1: [5, 6, 7, 8, 9],
    2: [10, 11, 12, 13, 14],
    3: [15, 16, 17, 18, 19],
    4: [20, 21, 22, 23, 24],
    5: [25, 26, 27, 28, 29],
    6: [30, 31, 32, 33, 34],
    7: [35, 36, 37, 38, 39],
    8: [40, 41, 42, 43, 44],
    9: [45, 46, 47, 48, 49],
}
capabilities: Dict[int | str, List[int | str]] = {
    0: [5, 6, 7, 8, 9, 10, 11],
    1: [0, 1, 2, 12, 13, 14],
    2: [15, 16, 17, 18, 19, 20],
    3: [21, 22, 23, 24, 25, 26],
    4: [27, 28, 29, 30, 31],
    5: [32, 33, 34, 35, 36, 37],
    6: [38, 39, 40, 41, 42],
    7: [43, 44, 45, 46, 47],
    8: [48, 49, 0, 1, 2],
    9: [3, 4, 5, 6, 7],
}
worker_availability: Dict[Tuple[int | str, str], bool] = {}

for w in workers:
    for s in shifts:
        # About 85% availability chance per worker per shift
        worker_availability[(w, s)] = random.random() < 0.85

solution: Dict[Tuple[int | str, int | str, str], bool] = solve_assignment(workers, tasks, shifts, demand, specialties, capabilities, worker_availability, True)

print("")

printTable(solution, workers, tasks, shifts)
