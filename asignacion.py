from collections import defaultdict

from ortools.sat.cp_model_pb2 import CpSolverStatus
from ortools.sat.python import cp_model
from ortools.sat.python.cp_model import CpModel, IntVar, LinearExpr, CpSolver

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


def calcular_puntuacion(
    bonus_maximo_jornada: int,
    capacidad_base: int,
    capacidad_decaimiento: int,
    especialidades: int,
    jornadas_noche: int,
    maximo_bonus_especialidad: int,
    penalizacion_no_voluntario_noche: int,
    preferencia_manana: list[Trabajador],
    preferencia_tarde: list[Trabajador],
    set_preferencia_manana: set[Trabajador],
    set_preferencia_tarde: set[Trabajador],
    set_voluntarios_noche: set[Trabajador],
    sets_especialidades: dict[PuestoTrabajo, set[Trabajador]],
    vars: dict[tuple[Trabajador, PuestoTrabajo, Jornada], IntVar],
    voluntarios_noche: list[Trabajador]
):
    # Se precomputan diccionarios mapeando cada trabajador a su posición en la lista_especialidades de voluntarios de noche, preferencia
    # de mañana y tarde y en cada lista_especialidades de especialidades, para mayor eficiencia posteriormente.
    index_voluntarios_noche: dict[Trabajador, int] = {trabajador: indice for indice, trabajador in
                                                      enumerate(voluntarios_noche)}
    index_preferencia_manana: dict[Trabajador, int] = {trabajador: indice for indice, trabajador in
                                                       enumerate(preferencia_manana)}
    index_preferencia_tarde: dict[Trabajador, int] = {trabajador: indice for indice, trabajador in
                                                      enumerate(preferencia_tarde)}
    index_especialidad: dict[PuestoTrabajo, dict[Trabajador, int]] = {
        puesto: {trabajador: indice for indice, trabajador in enumerate(lista)}
        for puesto, lista in especialidades.items()  # type: Trabajador, list[PuestoTrabajo]
    }
    # Se define un diccionario indexado por tuplas (trabajador, puesto, jornada) que representa la puntuación por
    # asignar ese trabajador a ese puesto en esa jornada.
    puntuaciones: dict[tuple[Trabajador, PuestoTrabajo, Jornada], int] = {}
    for trabajador, puesto, jornada in vars:  # type: Trabajador, PuestoTrabajo, Jornada

        # Puntuación por capacidad.
        puntuacion_capacidad = max(0, capacidad_base - capacidad_decaimiento * (trabajador.capacidades[puesto].id - 1))

        # Puntuación por estar más alto en las listas de especialidades.
        puntuacion_especialidad = 0
        if trabajador in sets_especialidades.get(puesto, set()):
            puntuacion_especialidad = max(0, maximo_bonus_especialidad - index_especialidad.get(puesto, {}).get(trabajador, maximo_bonus_especialidad))

        # Puntuación por preferencia de jornada o penalización por no ser voluntario para noche
        puntuacion_jornada = 0
        if jornada not in jornadas_noche:
            if trabajador in set_voluntarios_noche:
                # Si asignamos un voluntario de noche a una jornada de noche, obtenemos un bonus, proporcional a su
                # posición en la lista_especialidades.
                puntuacion_jornada = max(0, bonus_maximo_jornada - index_voluntarios_noche.get(trabajador, bonus_maximo_jornada))
            else:
                # Si asignamos a un no-voluntario a una jornada de noche, obtenemos una penalización.
                puntuacion_jornada -= penalizacion_no_voluntario_noche  # Penalty for assigning non-volunteer to a night shift
        elif (jornada.tipo_jornada == TipoJornada.MANANA) and trabajador in set_preferencia_manana:
            # Si asignamos a un trabajador con preferencia de mañana a una jornada de mañana, obtenemos un bonus,
            # proporcional a su posición en la lista_especialidades.
            puntuacion_jornada = max(0, bonus_maximo_jornada - index_preferencia_manana.get(trabajador,
                                                                                            bonus_maximo_jornada))
        elif (jornada.tipo_jornada == TipoJornada.TARDE) and trabajador in set_preferencia_tarde:
            # Si asignamos a un trabajador con preferencia de tarde a una jornada de tarde, obtenemos un bonus,
            # proporcional a su posición en la lista_especialidades.
            puntuacion_jornada = max(0, bonus_maximo_jornada - index_preferencia_tarde.get(trabajador, bonus_maximo_jornada))

        puntuaciones[trabajador, puesto, jornada] = puntuacion_capacidad + puntuacion_especialidad + puntuacion_jornada
    puntuacion_total: LinearExpr = LinearExpr.sum([
        puntuaciones[trabajador, puesto, jornada] * vars[trabajador, puesto, jornada]
        for trabajador, puesto, jornada in vars  # type: Trabajador, PuestoTrabajo, Jornada
    ])
    return puntuacion_total


def realizar_asignacion(
    trabajadores: list[Trabajador],
    puestos: list[PuestoTrabajo],
    jornadas: list[Jornada],
    # (puesto, jornada) -> demanda de trabajadores para ese puesto en esa jornada.
    demanda: dict[tuple[PuestoTrabajo, Jornada], int],
    # puesto -> lista_especialidades de trabajadores que tienen esa especialidad
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
    verbose_general: bool = False,
    verbose_asignacion_trabajadores: bool = False,
    verbose_asignacion_puestos: bool = False,
    # Parámetros de los pesos para la función objetivo a maximizar.
    capacidad_base: int = 100,
    capacidad_decaimiento: int = 10,
    maximo_bonus_especialidad: int = 50,
    bonus_maximo_jornada: int = 20,
    penalizacion_no_voluntario_noche: int = 100,

    forzar_maximo_asignaciones_especialidad: bool = False
) -> set[tuple[Trabajador, PuestoTrabajo, Jornada]]:

    model: CpModel = cp_model.CpModel()

    jornadas_no_puede_doblar: set[Jornada] = {
        jornada
        for jornada in jornadas # type: Jornada
        if not jornada.puede_doblar
    }

    jornadas_noche: set[Jornada] = {
        jornada
        for jornada in jornadas # type: Jornada
        if jornada.tipo_jornada == TipoJornada.NOCHE
    }

    # Se precomputan varios sets para comprobaciones de membresía más rápidas; O(1) en sets contra O(n) en listas.
    # No se pueden recibir los datos como sets directamente porque el orden es importante para otras cosas.
    set_voluntarios_noche: set[Trabajador] = set(voluntarios_noche)
    set_voluntarios_doble: set[Trabajador] = set(voluntarios_doble)
    set_preferencia_manana: set[Trabajador] = set(preferencia_manana)
    set_preferencia_tarde: set[Trabajador] = set(preferencia_tarde)
    sets_especialidades: dict[PuestoTrabajo, set[Trabajador]] = {
        puesto : set(lista_trabajadores)
        for puesto, lista_trabajadores in especialidades.items() # type: PuestoTrabajo, list[Trabajador]
    }

    # Se guardan las variables en un diccionario indexado por tuplas (trabajador, puesto, jornada)
    vars: dict[tuple[Trabajador, PuestoTrabajo, Jornada], IntVar] = {}

    # Se precomputan también diccionarios de listas de variables para mayor eficiencia posteriormente.
    vars_por_trabajador: dict[Trabajador, list[IntVar]] = defaultdict(list)
    vars_por_trabajador_y_jornada: dict[tuple[Trabajador, Jornada], list[IntVar]] = defaultdict(list)
    vars_por_puesto_y_jornada: dict[tuple[PuestoTrabajo, Jornada], list[IntVar]] = defaultdict(list)
    vars_trabajador_no_doblar: dict[Trabajador, list[IntVar]] = defaultdict(list)

    # Se crea una lista_especialidades de variables que representan la asignación de un trabajador a su especialidad.
    asignaciones_especialidades: list[IntVar] = []
    voluntario_a_noche: list[IntVar] = []

    for trabajador in trabajadores: # type: Trabajador
        for puesto in puestos: # type: PuestoTrabajo
            for jornada in jornadas: # type: Jornada
                # Solo se crean variables para las asignaciones permitidas, es decir, en las que el trabajador es
                # capaz de realizar el puesto y está disponible para esa jornada.
                #
                # Esta elección ayuda a la eficiencia del método, pero hace que no se pueda escribir alegremente
                # vars[trabajador, puesto, jornada], o nos arriesgamos a lanzar una excepción por intentar acceder
                # a un índice inexistente. En su lugar, habrá que usar vars.get((trabajador, puesto, jornada), 0)
                if puesto in trabajador.capacidades and (trabajador, jornada) in disponibilidad:
                    var = model.NewBoolVar(f'x_{trabajador}_{puesto}_{jornada}')
                    vars[trabajador, puesto, jornada] = var
                    vars_por_trabajador[trabajador].append(var)
                    vars_por_trabajador_y_jornada[trabajador, jornada].append(var)
                    vars_por_puesto_y_jornada[puesto, jornada].append(var)
                    if jornada in jornadas_no_puede_doblar:
                        vars_trabajador_no_doblar[trabajador].append(var)
                    if puesto in trabajador.especialidades:
                        asignaciones_especialidades.append(var)
                    if trabajador in set_voluntarios_noche and jornada in jornadas_noche:
                        voluntario_a_noche.append(var)

    num_trabajadores_disponibles: int = len({trabajador for (trabajador, puesto, jornada) in vars})
    total_asignaciones_especialidades: LinearExpr = LinearExpr.sum(asignaciones_especialidades)

    for trabajador in trabajadores: # type: Trabajador
        total_jornadas_trabajadas: LinearExpr = LinearExpr.sum(vars_por_trabajador[trabajador])
        if trabajador in set_voluntarios_doble:
            # Cada trabajador puede trabajar a lo sumo 2 jornadas, o 1 si no está en la lista_especialidades de voluntarios para dobles.
            model.Add(total_jornadas_trabajadas <= 2)
            for jornada in jornadas: # type: Jornada
                # Cada trabajador solo puede desempeñar un puesto en cada jornada, no se pueden dividir en dos.
                model.Add(LinearExpr.sum(vars_por_trabajador_y_jornada[trabajador, jornada]) <= 1)
            # Se crea una variable nueva para representar si el trabajador en el que estamos actualmente iterando
            # realiza o no una jornada doble. Nótese que solo hacemos esto para los que son voluntarios para dobles.
            doble_jornada: IntVar = model.NewBoolVar(f'double_shift_{trabajador}')
            # Forzamos a que la variable doble_jornada sea verdadera cuando se trabajan 2 jornadas, y falsa en otro caso.
            model.Add(total_jornadas_trabajadas == 2).OnlyEnforceIf(doble_jornada)
            model.Add(total_jornadas_trabajadas != 2).OnlyEnforceIf(doble_jornada.Not())
            # Un trabajador que dobla jornadas solo puede hacerlo en las que está permitido (las de mañana y tarde)
            model.Add(LinearExpr.sum(vars_trabajador_no_doblar[trabajador]) == 0).OnlyEnforceIf(doble_jornada)
        else:
            # Si el trabajador no es voluntario para doble, solo podrá trabajar 1 jornada, sin más complicaciones.
            model.Add(total_jornadas_trabajadas <= 1)

    # Cada jornada debe cubrir su demanda.
    for puesto in puestos: # type: PuestoTrabajo
        for jornada in jornadas: # type: Jornada
            model.Add(LinearExpr.sum(vars_por_puesto_y_jornada[puesto, jornada]) == demanda.get((puesto, jornada), 0))

    puntuacion_total = calcular_puntuacion(bonus_maximo_jornada, capacidad_base, capacidad_decaimiento, especialidades,
                                           jornadas_noche, maximo_bonus_especialidad, penalizacion_no_voluntario_noche,
                                           preferencia_manana, preferencia_tarde, set_preferencia_manana, set_preferencia_tarde,
                                           set_voluntarios_noche, sets_especialidades, vars, voluntarios_noche)

    if forzar_maximo_asignaciones_especialidad:
        # En el primer paso se busca maximizar el número de asignaciones a especialidades.
        model.Maximize(total_asignaciones_especialidades)

        solver: CpSolver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 8
        solver.parameters.search_branching = cp_model.FIXED_SEARCH
        status: CpSolverStatus = solver.Solve(model)

        if verbose_estadisticas_avanzadas:
            print("Paso 1: Estadísticas avanzadas")
            print(f"Problema resuelto en: {format_duration(solver.wall_time)}")
            print(f"Conflictos: {solver.NumConflicts()}")
            print(f"Ramas: {solver.NumBranches()}\n")

        if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            if verbose_estadisticas_avanzadas: print("No se encontró ninguna solución en el paso 1.")
            return set()

        # Encontrado el número máximo de asignaciones de especialidades, se añade una nueva restricción al modelo de que
        # la solución que aceptemos tenga ese número de asignaciones a especialidades, y no nos conformemos con menos.
        model.Add(total_asignaciones_especialidades == int(solver.ObjectiveValue()))

    # Cambiamos ahora la función a maximizar por la suma total de puntuaciones recibidas por las asignaciones realizadas.
    # De esta forma escogemos de entre las soluciones con máximas asignaciones a especialidades la que maximice también
    # nuestra puntuación.
    model.Maximize(puntuacion_total)

    # Se fuerza a que considere primero las asignaciones a noche
    # model.AddDecisionStrategy(voluntario_a_noche, cp_model.CHOOSE_FIRST, cp_model.SELECT_MAX_VALUE)

    solver2: CpSolver = cp_model.CpSolver()
    solver2.parameters.num_search_workers = 8
    solver2.parameters.search_branching = cp_model.FIXED_SEARCH
    status: CpSolverStatus = solver2.solve(model)

    if verbose_estadisticas_avanzadas:
        print("Paso 2: Estadísticas avanzadas")
        print(f"Problema resuelto en: {format_duration(solver2.wall_time)}")
        print(f"Conflictos: {solver2.NumConflicts()}")
        print(f"Ramas: {solver2.NumBranches()}\n")

    resultado: set[tuple[Trabajador, PuestoTrabajo, Jornada]] = set()

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        # Guardar en un conjunto de tuplas el resultado, representado las asignaciones que finalmerte son realizadas.
        resultado = {
            (trabajador, puesto, jornada)
            for (trabajador, puesto, jornada) in vars
            if solver2.Value(vars[trabajador, puesto, jornada]) == 1
        }

        numero_asignaciones_por_trabajador: dict[Trabajador, int] = defaultdict(int)
        voluntarios_noche_asignados_a_noche: int = 0
        preferencia_manana_asignados_a_manana: int = 0
        preferencia_tarde_asignados_a_tarde: int = 0
        for trabajador, puesto, jornada in resultado: # type: Trabajador, PuestoTrabajo, Jornada
            numero_asignaciones_por_trabajador[trabajador] += 1
            if jornada in jornadas_noche and trabajador in set_voluntarios_noche:
                voluntarios_noche_asignados_a_noche += 1
            if jornada == Jornada.MANANA and trabajador in set_preferencia_manana:
                preferencia_manana_asignados_a_manana += 1
            if jornada == Jornada.TARDE and trabajador in set_preferencia_tarde:
                preferencia_tarde_asignados_a_tarde += 1

        trabajadores_asignados_dobles: set[Trabajador] = {
            trabajador
            for trabajador in trabajadores # type: Trabajador
            if numero_asignaciones_por_trabajador[trabajador] == 2
        }

        trabajadores_asignados: int = len({trabajador for trabajador, _, _ in resultado})

        if verbose_general:
            if status == cp_model.OPTIMAL:
                print('Solución óptima encontrada.')
            else:
                print('Solución factible encontrada. No se puede garantizar que sea óptima.')

            puestos_demandados: int = 0
            puestos_nocturnos_demandados: int = 0
            for (_, jornada), valor in demanda.items(): # type: tuple[PuestoTrabajo, Jornada], int
                if jornada in jornadas_noche:
                    puestos_nocturnos_demandados += valor
                puestos_demandados += valor

            numero_voluntarios_noche: int = len(voluntarios_noche)
            numero_preferencia_manana: int = len(preferencia_manana)
            numero_preferencia_tarde: int = len(preferencia_tarde)
            asignaciones_especialidades_alcanzadas: int = solver2.Value(total_asignaciones_especialidades)

            print(f"\nSe demandaron {puestos_nocturnos_demandados} puestos nocturnos y hay {numero_voluntarios_noche} voluntarios de noche.")
            print(f"Se asignaron {voluntarios_noche_asignados_a_noche} de {numero_voluntarios_noche} ({100 * float(voluntarios_noche_asignados_a_noche) / numero_voluntarios_noche:.2f}%) voluntarios de noche a turnos de noche.\n")

            print(f"Se asignaron {preferencia_manana_asignados_a_manana} de {numero_preferencia_manana} ({100 * float(preferencia_manana_asignados_a_manana / numero_preferencia_manana):.2f}%) trabajadores con preferencia de mañana a turnos de mañana")
            print(f"Se asignaron {preferencia_tarde_asignados_a_tarde} de {numero_preferencia_tarde} ({100 * float(preferencia_tarde_asignados_a_tarde / numero_preferencia_tarde):.2f}%) trabajadores con preferencia de tarde a turnos de tarde")

            print(f'Número de asignaciones de especialidad alcanzado: {asignaciones_especialidades_alcanzadas} de {puestos_demandados} ({100 * asignaciones_especialidades_alcanzadas / puestos_demandados:.2f}%)')
            print(f'Número de trabajadores asignados: {trabajadores_asignados} de {num_trabajadores_disponibles} ({100 * float(trabajadores_asignados)/num_trabajadores_disponibles:.2f}%)')
            print(f'Puntuación alcanzada: {int(solver2.ObjectiveValue())}\n')

        if verbose_asignacion_trabajadores:
            for trabajador, puesto, jornada in resultado: # type: Trabajador, PuestoTrabajo, Jornada
                realiza_doble: str = 'DOBLE' if trabajador in trabajadores_asignados_dobles else ''
                polivalencia: str = trabajador.capacidades[puesto].nombre_es
                preferencia: str = 'P.MAÑANA' if trabajador in set_preferencia_manana else 'P.TARDE' if trabajador in set_preferencia_tarde else ''
                voluntario_noche: str = 'V.NOCHE' if trabajador in set_voluntarios_noche else ''
                voluntario_doble: str = 'V.DOBLE' if trabajador in set_voluntarios_doble else ''
                print(
                    f"Trabajador {trabajador:<{3}} -> "
                    f"{puesto.nombre_es:<{21}} | "
                    f"{jornada.nombre_es:<{10}} "
                    f"{realiza_doble:<{10}}{polivalencia:<{25}}"
                    f"{preferencia:<{10}}"
                    f"{voluntario_noche:<{10}}"
                    f"{voluntario_doble:<{10}}"
                )

        if verbose_asignacion_puestos:
            for jornada in jornadas: # type: Jornada
                print(f'{jornada.nombre_es.capitalize()}:')

                for puesto in puestos: # type: PuestoTrabajo
                    trabajadores_demandados: int = demanda.get((puesto, jornada), 0)
                    if trabajadores_demandados != trabajadores_asignados:
                        print('**********************************************')
                        print(f'{puesto} en jornada {jornada}: MISMATCH!!!!')
                        print('**********************************************')
                    print(f'{puesto.nombre_es} (demanda: {trabajadores_demandados}) asignado a {trabajadores_asignados} trabajadores:')

                    for trabajador in trabajadores: # type: Trabajador
                        if (trabajador, puesto, jornada) in resultado:
                            print(f'\tTrabajador {trabajador:<3}', f'\t{trabajador.capacidades[puesto].nombre_es}')

        ultimo_codigo_asignado_por_especialidad: dict[PuestoTrabajo, int | None] = {
            puesto : max({trabajador.codigo for (trabajador, puesto, _) in resultado if trabajador in sets_especialidades.get(puesto, {})}, default=None)
            for puesto in puestos # type: PuestoTrabajo
        }

        trabajadores_voluntarios_asignados_noche: set[Trabajador] = {
            trabajador
            for trabajador, _, jornada in resultado  # type: Trabajador, PuestoTrabajo, Jornada
            if trabajador in set_voluntarios_noche
            and jornada in jornadas_noche
        }

        ultimo_codigo_voluntarios_noche: int | None = max({
            trabajador.codigo
            for trabajador in trabajadores_voluntarios_asignados_noche
        }, default=None)

        ultimo_codigo_voluntarios_doble: int | None = max({
            trabajador.codigo
            for trabajador in trabajadores_asignados_dobles # type: Trabajador
        }, default=None)

    elif verbose_estadisticas_avanzadas:
        print('Ninguna solución factible encontrada')

    return resultado
