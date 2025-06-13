from ortools.sat.python import cp_model
from ortools.sat.python.cp_model import CpModel
from typing import List, Optional

def solve_assignment_problem(
    efficiency: List[List[float]],
    demand: List[int],
    verbose: bool = False,
    global_efficiency_threshold: Optional[float] = None,
    task_efficiency_threshold: Optional[List[float]] = None,
) -> Optional[dict]:
    # Deduce the number of workers and tasks from the rows and columns of the efficiency matrix
    workers: int = len(efficiency) # Number of rows of the efficiency matrix
    tasks: int = len(efficiency[0]) if efficiency else 0 # Number of columns of the efficiency matrix
    demanded_workers: int = sum(demand) # Total number of workers demanded by the tasks

    if task_efficiency_threshold is not None and len(task_efficiency_threshold) != tasks:
        if verbose:
            print("The task efficiency threshold list's length doesn't match the number of tasks.")
            if len(task_efficiency_threshold) < tasks:
                print("The task efficiency threshold list will be padded with zeroes.")
            if len(task_efficiency_threshold) > tasks:
                print("The task efficiency threshold list will be trimmed from the end.")
        while len(task_efficiency_threshold) < tasks:
            task_efficiency_threshold.append(0)
        while len(task_efficiency_threshold) > tasks:
            task_efficiency_threshold.pop()

    if verbose and demanded_workers < workers:
        print(f'Warning: There are {workers} available workers but only {demanded_workers} are demanded.')
        print('At least one task will have more workers than needed on it.\n')

    if demanded_workers > workers:
        if verbose:
            print(f'Error: There are {workers} available workers but {demanded_workers} are demanded.')
            print('An optimal solution can never be found.')
        return {}

    if workers == 0 or tasks == 0:
        if verbose: print(f'Error: there are {workers} workers and {tasks} tasks.')
        return {}

    # Create the model, of type CpModel
    model: CpModel = cp_model.CpModel()

    # Create the assignment variables as a dictionary, indexed by tuples (worker, task)
    x: dict = {}
    for i in range(workers):
        for j in range(tasks):
            x[i, j] = model.NewBoolVar(f'x_{i}_{j}')

    # Constraint: each worker is assigned to exactly one task
    for i in range(workers):
        model.Add(sum(x[i, j] for j in range(tasks)) == 1)

    # Constraint: each task must be assigned at least its required number of workers
    for j in range(tasks):
        model.Add(sum(x[i, j] for i in range(workers)) >= demand[j])

    # Constraint: workers cannot be assigned to tasks they can't perform (efficiency == 0)
    for i in range(workers):
        for j in range(tasks):
            if (
                (global_efficiency_threshold is not None and efficiency[i][j] < global_efficiency_threshold)
                or (task_efficiency_threshold is not None and efficiency[i][j] < task_efficiency_threshold[j])
                or efficiency[i][j] == 0
            ):
                model.Add(x[i, j] == 0)


    # Objective: maximize total efficiency
    model.Maximize(
        sum(efficiency[i][j] * x[i, j] for i in range(workers) for j in range(tasks))
    )

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    # print info about the resolution process
    if verbose:
        print("Advanced usage:")
        print(f"Problem solved in {solver.wall_time} seconds")
        print(f"Conflicts: {solver.NumConflicts()}")
        print(f"Branches: {solver.NumBranches()}\n")

    # Return the solution if an optimal one was found, or {} otherwise.
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        result = {}
        for i in range(workers):
            for j in range(tasks):
                result[i, j] = solver.Value(x[i, j])
        # Print info about the found solution.
        if verbose:
            if status == cp_model.OPTIMAL:
                print('Optimal solution found.')
            else:
                print('Feasible solution found. Optimality cannot be guaranteed.')

            print(f'Total efficiency = {solver.ObjectiveValue()}\n')
            for i in range(workers):
                for j in range(tasks):
                    if result[i, j]:
                        print(f'Worker {i} assigned to Task {j} with efficiency {efficiency[i][j]}')
            print("")
            for j in range(tasks):
                print(f'Task {j} assigned to:')
                for i in range(workers):
                    if result[i, j]:
                        print(f'\tWorker {i} with efficiency {efficiency[i][j]}')
        return result
    else:
        print('No feasible solution found.')
        return {}

efficiency: List[List[float]] = [
    [0.82, 0.71, 0.65, 0,    0.93, 0.58, 0,    0.89, 0.74, 0.60],
    [0.45, 0.69, 0.81, 0.72, 0,    0.66, 0.77, 0,    0.61, 0.56],
    [0,    0.79, 0.91, 0,    0.74, 0.68, 0.59, 0.65, 0,    0.72],
    [0.91, 0.62, 0,    0.88, 0.93, 0,    0.71, 0.67, 0.59, 0],
    [0.68, 0,    0.55, 0.81, 0.77, 0.72, 0,    0.88, 0.66, 0.49],
    [0.87, 0.65, 0.78, 0.61, 0,    0,    0.70, 0.75, 0.62, 0.58],
    [0,    0.92, 0.84, 0.73, 0.67, 0,    0.89, 0.71, 0,    0.64],
    [0.59, 0.74, 0,    0.85, 0.60, 0.69, 0.72, 0,    0.73, 0.57],
    [0.67, 0.81, 0.91, 0.79, 0,    0.75, 0.64, 0.68, 0,    0.53],
    [0,    0.85, 0.74, 0.61, 0.78, 0.59, 0.66, 0,    0.57, 0.69],
    [0.76, 0.59, 0.66, 0.87, 0.61, 0.78, 0.60, 0,    0.55, 0],
    [0.83, 0.70, 0.73, 0.64, 0.72, 0.66, 0.74, 0.62, 0.71, 0.69],
    [0.54, 0,    0.88, 0.75, 0.85, 0.77, 0,    0.73, 0.70, 0.65],
    [0.77, 0.62, 0.71, 0.89, 0.69, 0.60, 0.55, 0.58, 0.67, 0.72],
    [0,    0.66, 0.70, 0.82, 0.64, 0.68, 0.73, 0,    0.76, 0.74],
    [0.62, 0.60, 0.67, 0.76, 0,    0.73, 0.61, 0.80, 0.66, 0.59],
    [0.70, 0.64, 0,    0.69, 0.71, 0.67, 0.75, 0.65, 0.60, 0.58],
    [0.85, 0.77, 0.72, 0.63, 0.68, 0,    0.80, 0.66, 0.79, 0.73],
    [0.81, 0,    0.90, 0.71, 0.73, 0.69, 0.59, 0.74, 0.68, 0.76],
    [0.73, 0.78, 0.61, 0.66, 0.75, 0.70, 0.77, 0.72, 0.63, 0],
    [0.66, 0.61, 0.79, 0.70, 0.76, 0.74, 0,    0.69, 0.65, 0.60],
    [0.88, 0.74, 0.63, 0.60, 0.70, 0.65, 0.71, 0.67, 0.62, 0.64],
    [0.75, 0.80, 0.69, 0.72, 0.66, 0.63, 0.78, 0,    0.70, 0.73],
    [0.60, 0.57, 0.76, 0.65, 0.59, 0.71, 0.69, 0.63, 0.67, 0.62],
    [0.64, 0.72, 0.75, 0.67, 0.62, 0.61, 0.74, 0.70, 0.66, 0],
    [0.58, 0.68, 0.73, 0.59, 0.65, 0.64, 0.70, 0.68, 0.69, 0.71],
    [0.71, 0.63, 0.70, 0.66, 0.60, 0.69, 0.76, 0.61, 0.64, 0],
    [0.80, 0.60, 0.68, 0.73, 0.57, 0.66, 0.79, 0.64, 0.62, 0.58],
    [0.74, 0.67, 0.62, 0.58, 0.61, 0.72, 0.68, 0.66, 0.65, 0.63],
    [0.69, 0.65, 0.77, 0.60, 0.58, 0.68, 0.67, 0.62, 0.61, 0.59],
    [0.72, 0.69, 0.74, 0.62, 0.63, 0.70, 0.65, 0.67, 0.60, 0.66]
]

demand: List[int] = [4, 3, 5, 4, 3, 3, 2, 3, 2, 2]

task_efficiency_threshold: List[float] = [0.5, 0.4, 0.6, 0.7, 0.7, 0.65, 0.55, 0.4, 0.55, 0.7]

assignment = solve_assignment_problem(efficiency, demand, True, 0.5, task_efficiency_threshold)