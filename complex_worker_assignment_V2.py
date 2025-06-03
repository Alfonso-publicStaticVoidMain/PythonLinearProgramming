import random
from pickle import FALSE

from ortools.sat.python import cp_model
from ortools.sat.python.cp_model import CpModel, IntVar
from typing import List, Optional, Dict, Tuple


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


def solve_assignment(
        # Names or IDs of the workers.
        workers: List[int | str],

        # Names or identifiers of the tasks.
        tasks: List[int | str] | Dict[int, str],

        # Names of the shifts. For now, they are assumed ['morning', 'afternoon', 'night_1', 'night_2']
        shifts: List[int | str] | Dict[int, str],

        # For each task and shift, maps it to the number of workers demanded for it.
        demand: Dict[Tuple[int | str, str], int],

        # Maps each tuple of worker and task to their capability on that task, 1 being their main specialty and higher numbers meaning less capability.
        # If a pair of (worker, task) is unassigned in worker_capabilities, it should be assumed the worker is incapable of performing the task.
        worker_capabilities: Dict[Tuple[int | str, int | str], int],

        # Maps each task to the List of the workers that have it as their specialty, ordered by their priority of assignment
        specialties: Dict[int | str, List[int | str]],

        # Workers that are volunteers for night shifts, ordered. They will be chosen for night shifts over others if able
        night_volunteers: List[int | str],

        # Workers that are preferred for morning shifts, ordered.
        morning_preference: List[int | str],  # Contains the workers which are preferred for morning shifts, ordered.

        # Workers that are preferred for afternoon shifts, ordered.
        afternoon_preference: List[int | str],

        # If a pair (worker, shift) is present, that worker is available for that shift.
        worker_availability: List[Tuple[int | str, str]],

        # Workers that can perform double shifts.
        double_shift_availability: List[int | str],

        # State parameter to print the result via console when set to True.
        verbose: bool = False,

        # Parameters for calculating the scores of the assignments, with their default values
        capability_base: int = 100,
        capability_decay: int = 10,
        specialty_bonus_max: int = 50,
        shift_bonus_max: int = 20,
        night_shift_penalty: int = 30,
) -> Dict[Tuple[int | str, int | str, str], bool]:
    model: CpModel = cp_model.CpModel()

    # Define the variables and store them on a dictionary
    vars: Dict[Tuple[int | str, int | str, str], IntVar] = {}
    for w in workers:
        for t in tasks:
            for s in shifts:
                # Only create a variable if the worker can perform the task and is available in that shift
                if (w, t) in worker_capabilities and (w, s) in worker_availability:
                    vars[w, t, s] = model.NewBoolVar(f'x_{w}_{t}_{s}')
                # This choice was made for performance and to avoid creating variables that will all be later set to 0.
                # Because of it we can't just call vars[w, t, s] recklessly or an exception will be thrown.
                # We instead have to use vars.get((w, t, s), 0), which returns the value for the (w, t, s) key, or
                # 0 if that key is absent from the dictionary.

    for w in workers:
        # Each worker can have at most 2 shifts, or 1 if they are not in the double shift list
        if w in double_shift_availability:
            model.Add(sum(vars.get((w, t, s), 0) for t in tasks for s in shifts) <= 2)
            for s in shifts:
                # Each worker can work at most 1 task each shift
                model.Add(sum(vars.get((w, t, s), 0) for t in tasks) <= 1)
        else:
            model.Add(sum(vars.get((w, t, s), 0) for t in tasks for s in shifts) <= 1)

        # A worker that works double shifts mustn't do it during night

        # A bool var is created, representing if a double shift was done
        is_double_shift: IntVar = model.NewBoolVar(f'double_shift_{w}')
        total_shifts: int = sum(vars.get((w, t, s), 0) for t in tasks for s in shifts)
        model.Add(total_shifts == 2).OnlyEnforceIf(is_double_shift)

        night_shifts = sum(vars.get((w, t, s), 0) for t in tasks for s in ['night_1', 'night_2'])
        model.Add(night_shifts == 0).OnlyEnforceIf(is_double_shift)

    # Each task must meet its demand exactly on each shift
    for t in tasks:
        for s in shifts:
            model.Add(sum(vars.get((w, t, s), 0) for w in workers) == demand[t, s])

    # Define the score of assigning a worker to a task on a shift, according to its position on the lists and if it's their specialty
    scores: Dict[Tuple[int | str, int | str, str], int] = {}
    for (w, t, s) in vars:
        # Capability score
        capability_score = max(0, capability_base - capability_decay * (worker_capabilities[w, t] - 1))
        # Specialty score
        specialty_list = specialties.get(t, [])
        specialty_score = 0
        if w in specialty_list:
            specialty_rank = specialty_list.index(w)
            specialty_score = max(0, specialty_bonus_max - specialty_rank)
        # Shift preference or penalty
        shift_bonus = 0
        if s.startswith("night"):
            if w in night_volunteers:
                shift_bonus = max(0, shift_bonus_max - night_volunteers.index(w))
            else:
                shift_bonus -= night_shift_penalty  # Penalty for assigning non-volunteer to a night shift
        elif s == "morning" and w in morning_preference:
            shift_bonus = max(0, shift_bonus_max - morning_preference.index(w))
        elif s == "afternoon" and w in afternoon_preference:
            shift_bonus = max(0, shift_bonus_max - afternoon_preference.index(w))
        scores[(w, t, s)] = capability_score + specialty_score + shift_bonus

    # Create a List of variables where it represents the assignment of a worker into its specialty
    specialties_assigned: List[IntVar] = []
    for (w, t, s), var in vars.items():
        if w in specialties[t]:
            specialties_assigned.append(var)

    # Set the first step of the model to maximize specialty assignments
    model.Maximize(sum(specialties_assigned))

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
    model.Add(sum(specialties_assigned) == int(solver.ObjectiveValue()))

    # Set the new function to maximize, a sum of the assignments pondered by their scores
    model.Maximize(sum(scores[w, t, s] * vars[w, t, s] for (w, t, s) in vars))

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
                    result[w, t, s] = bool(solver.Value(vars[w, t, s])) if (w, t, s) in vars else False

        if verbose:
            if status == cp_model.OPTIMAL:
                print('Optimal solution found.')
            else:
                print('Feasible solution found. Optimality cannot be guaranteed.')

            print(f'Number of specialty assignments achieved: {int(solver.ObjectiveValue())} out of {sum(demand.get((t, s), 0) for t in tasks for s in shifts)}')
            print(f'Number of workers assigned: {len({w for (w, t, s) in result})} out of {len(workers)}\n')

            for (w, t, s) in result:
                if result[w, t, s]:
                    print(f'Worker {w} assigned to Task {t} on the {s} shift')

            print("")

            for t in tasks:
                for s in shifts:
                    print(f'Task {t} on {s} shift assigned to:')
                    for w in workers:
                        if result[w, t, s]:
                            print(f'\tWorker {w}')

    elif verbose: print('No feasible solution found.')

    return result


def generate_random_parameters(
        num_workers: int,
        num_tasks: int,
        chance_to_be_available: float = 0.85
) -> Tuple[
    List[int],  # workers
    List[int],  # tasks
    List[str],  # shifts
    Dict[Tuple[int, str], int],  # demand
    Dict[Tuple[int, int], int],  # worker_capabilities
    Dict[int, List[int]],  # specialties
    List[int],  # night_volunteers
    List[int],  # morning_preference
    List[int],  # afternoon_preference
    List[Tuple[int, str]],  # worker_availability
    List[int]  # double_shift_availability
]:
    workers = list(range(num_workers))
    tasks = list(range(num_tasks))
    shifts = ['morning', 'afternoon', 'night_1', 'night_2']

    # Demand
    demand = {}
    total_demand = int(num_workers * random.uniform(1.25, 1.33))
    base_demand_per_task_shift = total_demand // (num_tasks * len(shifts))
    for t in tasks:
        for s in shifts:
            demand[(t, s)] = base_demand_per_task_shift + random.randint(-2, 2)

    # Capabilities and specialties
    worker_capabilities = {}
    specialties = {t: [] for t in tasks}

    for w in workers:
        specialty = random.choice(tasks)
        worker_capabilities[w, specialty] = 1
        specialties[specialty].append(w)

        for _ in range(random.randint(0, 3)):
            t = random.choice(tasks)
            if (w, t) not in worker_capabilities:
                worker_capabilities[w, t] = random.randint(2, 5)

    for t in specialties:
        random.shuffle(specialties[t])

    # Availability
    worker_availability = [(w, s) for w in workers for s in shifts if random.random() < chance_to_be_available]

    # Preferences
    half = num_workers // 2
    morning_preference = random.sample(workers, k=half)
    afternoon_preference = list(set(workers) - set(morning_preference))
    night_volunteers = random.sample(workers, k=random.randint(num_workers // 6, num_workers // 4))

    double_shift_availability = random.sample(workers, k=int(num_workers * 0.6))

    return (
        workers,
        tasks,
        shifts,
        demand,
        worker_capabilities,
        specialties,
        night_volunteers,
        morning_preference,
        afternoon_preference,
        worker_availability,
        double_shift_availability
    )


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

#solution: Dict[Tuple[int | str, int | str, str], bool] = solve_assignment(*generate_random_parameters(300, 15), verbose=True)
