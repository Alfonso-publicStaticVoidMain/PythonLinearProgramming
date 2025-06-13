from ortools.sat.python import cp_model
from ortools.sat.python.cp_model import CpModel, IntVar

from Clases import Trabajador, PuestoTrabajo, Jornada, NivelDesempeno, TipoJornada


def format_duration(seconds: float | int) -> str:
    if seconds < 10:
        return f"{seconds:.4f} segundos"
    seconds_int = int(seconds)
    if seconds_int < 60:
        return f"{seconds_int} segundos"
    elif seconds_int < 3600:
        minutes = seconds_int // 60
        rem_seconds = seconds_int % 60
        return f"{minutes} minutos y {rem_seconds} segundos"
    else:
        hours = seconds_int // 3600
        rem_seconds = seconds_int % 3600
        minutes = rem_seconds // 60
        rem_seconds = rem_seconds % 60
        return f"{hours} horas, {minutes} minutos y {rem_seconds} segundos"


def realizar_asignacion(
    trabajadores: list[Trabajador],
    puestos: list[PuestoTrabajo],
    jornadas: list[Jornada],
    # (puesto, jornada) -> demanda de trabajadores para ese puesto en esa jornada.
    demanda: dict[tuple[PuestoTrabajo, Jornada], int],
    # puesto -> lista de trabajadores que tienen esa especialidad
    especialidades: dict[PuestoTrabajo, list[Trabajador]],
    # Lista de voluntarios de noche, ordenados por preferencia
    voluntarios_noche: list[Trabajador],
    # Lista de pares (trabajador, jornada) en los que el trabajador está disponible en esa jornada
    disponibilidad: set[tuple[Trabajador, Jornada]],
    # Lista de voluntarios para dobles, ordenados por preferencia
    voluntarios_doble: list[Trabajador],
    # Lista de preferencia de mañana, ordenados por preferencia
    preferencia_manana: list[Trabajador],
    # Lista de preferencia de tarde, ordenados por preferencia
    preferencia_tarde: list[Trabajador],

    # Parámetros para controlar si se imprime por pantalla la solución encontrada y estadísticas sobre la resolución.
    verbose_estadisticas_avanzadas: bool = False,
    verbose_asignacion_trabajadores: bool = False,
    verbose_asignacion_puestos: bool = False,
    # Parámetros de los pesos para la función objetivo a maximizar.
    capacidad_base: int = 100,
    capacidad_decaimiento: int = 10,
    maximo_bonus_especialidad: int = 50,
    bonus_maximo_jornada: int = 20,
    penalizacion_no_voluntario_noche: int = 50,
) -> dict[tuple[Trabajador, PuestoTrabajo, Jornada], bool]:

    model: CpModel = cp_model.CpModel()
    jornadas_puede_doblar: set[Jornada] = {jornada for jornada in jornadas if jornada.puede_doblar}
    jornadas_noche: set[Jornada] = {jornada for jornada in jornadas if jornada.tipo_jornada == TipoJornada.NOCHE}

    # Se precomputan varios conjuntos para comprobaciones de membresía más rápidas; O(1) en sets contra O(n) en listas.
    set_voluntarios_noche: set[Trabajador] = set(voluntarios_noche)
    set_voluntarios_doble: set[Trabajador] = set(voluntarios_doble)
    set_preferencia_manana: set[Trabajador] = set(preferencia_manana)
    set_preferencia_tarde: set[Trabajador] = set(preferencia_tarde)
    sets_especialidades: dict[PuestoTrabajo, set[Trabajador]] = {}
    for puesto, lista_trabajadores in especialidades.items():
        sets_especialidades[puesto] = set(lista_trabajadores)

    # Define the variables and store them on a dictionary indexed by tuples of (worker, task, shift)
    vars: dict[tuple[Trabajador, PuestoTrabajo, Jornada], IntVar] = {}
    for trabajador in trabajadores:
        for puesto in puestos:
            for jornada in jornadas:
                # Solo se crean variables para las asignaciones permitidas, es decir, en las que el trabajador es
                # capaz de realizar el puesto y está disponible para esa jornada.
                #
                # Esta elección ayuda a la eficiencia del método, pero hace que no se pueda escribir alegremente
                # vars[trabajador, puesto, jornada], o nos arriesgamos a lanzar una excepción por intentar acceder
                # a un índice inexistente. En su lugar, habrá que usar vars.get((trabajador, puesto, jornada), 0)
                if puesto in trabajador.capacidades and (trabajador, jornada) in disponibilidad:
                    vars[trabajador, puesto, jornada] = model.NewBoolVar(f'x_{trabajador}_{puesto}_{jornada}')

    num_trabajadores_disponibles = len({trabajador for (trabajador, puesto, jornada) in vars})

    for trabajador in trabajadores:
        # Cada trabajador puede trabajar a lo sumo 2 jornadas, o 1 si no está en la lista de voluntarios para dobles.
        if trabajador in set_voluntarios_doble:
            total_jornadas_trabajadas: int = sum(vars.get((trabajador, t, s), 0) for t in puestos for s in jornadas)
            model.Add(total_jornadas_trabajadas <= 2)
            for jornada in jornadas:
                # Cada trabajador solo puede desempeñar un puesto en cada jornada, no se pueden dividir en dos.
                model.Add(sum(vars.get((trabajador, puesto, jornada), 0) for puesto in puestos) <= 1)
            # Se crea una variable nueva para representar si el trabajador en el que estamos actualmente iterando
            # realiza o no una jornada doble. Nótese que solo hacemos esto para los que son voluntarios para dobles.
            doble_jornada: IntVar = model.NewBoolVar(f'double_shift_{trabajador}')
            # Forzamos a que la variable doble_jornada sea verdadera cuando se trabajan 2 jornadas, y falsa en otro caso.
            model.Add(total_jornadas_trabajadas == 2).OnlyEnforceIf(doble_jornada)
            model.Add(total_jornadas_trabajadas != 2).OnlyEnforceIf(doble_jornada.Not())
            # Un trabajador que dobla jornadas solo puede hacerlo en las que está permitido (las de mañana y tarde)
            model.Add(sum(vars.get((trabajador, puesto, jornada), 0) for puesto in puestos for jornada in jornadas_puede_doblar) == 0).OnlyEnforceIf(doble_jornada)
        else:
            # Si el trabajador no es voluntario para doble, solo podrá trabajar 1 jornada, sin más complicaciones.
            model.Add(sum(vars.get((trabajador, puesto, jornada), 0) for puesto in puestos for jornada in jornadas) <= 1)

    # Cada jornada debe cubrir su demanda.
    for puesto in puestos:
        for jornada in jornadas:
            model.Add(sum(vars.get((trabajador, puesto, jornada), 0) for trabajador in trabajadores) == demanda.get((puesto, jornada), 0))

    # Se define un diccionario indexado por tuplas (trabajador, puesto, jornada) que representa la puntuación por
    # asignar ese trabajador a ese puesto en esa jornada.
    puntuaciones: dict[tuple[Trabajador, PuestoTrabajo, Jornada], int] = {}
    for (trabajador, puesto, jornada) in vars:

        # Puntuación por capacidad.
        puntuacion_capacidad = max(0, capacidad_base - capacidad_decaimiento * (trabajador.capacidades[puesto].id - 1))

        # Puntuación por estar más alto en las listas de especialidades.
        lista_especialidad = especialidades.get(puesto, [])
        set_especialidad = sets_especialidades.get(puesto, set())
        puntuacion_especialidad = 0
        if trabajador in set_especialidad:
            puntuacion_especialidad = max(0, maximo_bonus_especialidad - lista_especialidad.index(trabajador))

        # Puntuación por preferencia de jornada o penalización por no ser voluntario para noche
        puntuacion_jornada = 0
        if jornada not in jornadas_noche:
            if trabajador in set_voluntarios_noche:
                # Si asignamos un voluntario de noche a una jornada de noche, obtenemos un bonus, proporcional a su
                # posición en la lista.
                puntuacion_jornada = max(0, bonus_maximo_jornada - voluntarios_noche.index(trabajador))
            else:
                # Si asignamos a un no-voluntario a una jornada de noche, obtenemos una penalización.
                puntuacion_jornada -= penalizacion_no_voluntario_noche  # Penalty for assigning non-volunteer to a night shift
        elif (jornada.tipo_jornada == TipoJornada.MANANA) and trabajador in set_preferencia_manana:
            # Si asignamos a un trabajador con preferencia de mañana a una jornada de mañana, obtenemos un bonus,
            # proporcional a su posición en la lista.
            puntuacion_jornada = max(0, bonus_maximo_jornada - preferencia_manana.index(trabajador))
        elif (jornada.tipo_jornada == TipoJornada.TARDE) and trabajador in set_preferencia_tarde:
            # Si asignamos a un trabajador con preferencia de tarde a una jornada de tarde, obtenemos un bonus,
            # proporcional a su posición en la lista.
            puntuacion_jornada = max(0, bonus_maximo_jornada - preferencia_tarde.index(trabajador))

        puntuaciones[trabajador, puesto, jornada] = puntuacion_capacidad + puntuacion_especialidad + puntuacion_jornada

    # Se crea una lista de variables que representan la asignación de un trabajador a su especialidad.
    asignaciones_especialidades: list[IntVar] = []
    for (trabajador, puesto, jornada), var in vars.items():
        if trabajador in especialidades[puesto]:
            asignaciones_especialidades.append(var)

    # En el primer paso del modelo se busca maximizar el número de asignaciones a especialidades.
    model.Maximize(sum(asignaciones_especialidades))

    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 8
    solver.parameters.search_branching = cp_model.FIXED_SEARCH
    status = solver.Solve(model)

    if verbose_estadisticas_avanzadas:
        print("Paso 1: Estadísticas avanzadas")
        print(f"Problema resuelto en: {format_duration(solver.wall_time)}")
        print(f"Conflictos: {solver.NumConflicts()}")
        print(f"Ramas: {solver.NumBranches()}\n")

    if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        if verbose_estadisticas_avanzadas: print("No se encontró ninguna solución en el paso 1.")
        return {}

    # Encontrado el número máximo de asignaciones de especialidades, se añade una nueva restricción al modelo de que
    # la solución que aceptemos tenga ese número de asignaciones a especialidades, y no nos conformemos con menos.
    model.Add(sum(asignaciones_especialidades) == int(solver.ObjectiveValue()))

    # Cambiamos ahora la función a maximizar por la suma total de puntuaciones recibidas por las asignaciones realizadas.
    # De esta forma escogemos de entre las soluciones con máximas asignaciones a especialidades la que maximice también
    # nuestra puntuación.
    model.Maximize(sum(puntuaciones[trabajador, puesto, jornada] * vars[trabajador, puesto, jornada] for (trabajador, puesto, jornada) in vars))

    solver2 = cp_model.CpSolver()
    solver2.parameters.num_search_workers = 8
    solver2.parameters.search_branching = cp_model.FIXED_SEARCH
    status = solver2.solve(model)

    if verbose_estadisticas_avanzadas:
        print("Paso 2: Estadísticas avanzadas")
        print(f"Problema resuelto en: {format_duration(solver2.wall_time)}")
        print(f"Conflictos: {solver2.NumConflicts()}")
        print(f"Ramas: {solver2.NumBranches()}\n")

    resultado: dict[tuple[Trabajador, PuestoTrabajo, Jornada], bool] = {}
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        # Guardar en un diccionario los resultados, mapeando las tuplas a True si se decide realizar la asignación, y
        # False en otro caso o si la variable asociada a esa tupla no fue considerada como asignación válida.
        resultado = {
            (trabajador, puesto, jornada) : ((trabajador, puesto, jornada) in vars and solver2.Value(vars[trabajador, puesto, jornada]) == 1)
            for trabajador in trabajadores
            for puesto in puestos
            for jornada in jornadas
        }

        trabajadores_asignados_dobles: list[Trabajador] = [
            trabajador
            for trabajador in trabajadores
            if sum(resultado[trabajador, p, j] for p in puestos for j in jornadas) == 2
        ]

        if verbose_asignacion_trabajadores:
            if status == cp_model.OPTIMAL:
                print('Solución óptima encontrada.')
            else:
                print('Solución factible encontrada. No se puede garantizar que sea óptima.')

            trabajadores_asignados: int = len({trabajador for (trabajador, puesto, jornada), val in resultado.items() if val})
            puestos_demandados: int = sum(demanda.get((puesto, jornada), 0) for puesto in puestos for jornada in jornadas)
            print(f'Número de asignaciones de especialidad alcanzado: {int(solver.ObjectiveValue())} de {puestos_demandados} ({100*solver.ObjectiveValue()/puestos_demandados:.2f}%)')
            print(f'Número de trabajadores asignados: {trabajadores_asignados} de {num_trabajadores_disponibles} ({100 * float(trabajadores_asignados) /num_trabajadores_disponibles:.2f}%)\n')

            for (trabajador, puesto, jornada) in resultado:
                if resultado[trabajador, puesto, jornada]:
                    double = 'DOBLE' if trabajador in trabajadores_asignados_dobles else ''
                    poly = trabajador.capacidades[puesto].nombre_es
                    print(
                        f"Trabajador {trabajador:<{3}} -> "
                        f"{puesto.nombre_es:<{21}} | "
                        f"{jornada.nombre_es:<{10}} "
                        f"{double:<{10}}{poly:<{25}}"
                    )

        if verbose_asignacion_puestos:
            for jornada in jornadas:
                print(f'{jornada.nombre_es.capitalize()}:')
                for puesto in puestos:
                    print(f'{puesto.nombre_es} asignado a:')
                    for trabajador in trabajadores:
                        if resultado[trabajador, puesto, jornada]:
                            print(f'\tTrabajador {trabajador:<3}', f'\t{trabajador.capacidades[puesto].nombre_es}')

    elif verbose_estadisticas_avanzadas:
        print('Ninguna solución factible encontrada')

    return resultado
