from collections import defaultdict

from ortools.sat.cp_model_pb2 import CpSolverStatus
from ortools.sat.python import cp_model
from ortools.sat.python.cp_model import CpModel, IntVar, LinearExpr, CpSolver

from Clases import Trabajador, PuestoTrabajo, Jornada, NivelDesempeno, TipoJornada
from parse import data
from SolutionHandler import SolutionHandler, soluciones


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


def print_estadisticas_avanzadas(solver: CpSolver, mensaje: str = ""):
    print(mensaje)
    print(f"Problema resuelto en: {format_duration(solver.wall_time)}")
    print(f"Conflictos: {solver.NumConflicts()}")
    print(f"Ramas: {solver.NumBranches()}\n")


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

    # Parámetros para ver que cosas maximizar
    maximizar_asignaciones_a_especialidad: bool = True,
    maximizar_asignaciones_voluntarios_noche: bool = True,
    maximizar_asignaciones_preferencias: bool = True
):

    model: CpModel = CpModel()

    jornadas_no_puede_doblar: set[Jornada] = {
        jornada
        for jornada in jornadas
        if not jornada.puede_doblar
    }

    jornadas_noche: set[Jornada] = {
        jornada
        for jornada in jornadas
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
        for puesto, lista_trabajadores in especialidades.items()
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
    asignacion_voluntario_noche_a_noche: list[IntVar] = []
    asignacion_preferencia_manana_a_manana: list[IntVar] = []
    asignacion_preferencia_tarde_a_tarde: list[IntVar] = []

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
                        asignacion_voluntario_noche_a_noche.append(var)
                    if trabajador in set_preferencia_manana and jornada == Jornada.MANANA:
                        asignacion_preferencia_manana_a_manana.append(var)
                    if trabajador in set_preferencia_tarde and jornada == Jornada.TARDE:
                        asignacion_preferencia_tarde_a_tarde.append(var)

    num_trabajadores_disponibles: int = len({trabajador for (trabajador, puesto, jornada) in vars})
    total_asignaciones_especialidades: LinearExpr = LinearExpr.sum(asignaciones_especialidades)
    vol_noche_asignados_a_noche: LinearExpr = LinearExpr.sum(asignacion_voluntario_noche_a_noche)
    pref_manana_asignados_a_manana: LinearExpr = LinearExpr.sum(asignacion_preferencia_manana_a_manana)
    pref_tarde_asignados_a_tarde: LinearExpr = LinearExpr.sum(asignacion_preferencia_tarde_a_tarde)
    jornadas_trabajadas_por_trabajador: dict[Trabajador, LinearExpr] = {}

    for trabajador in trabajadores:
        total_jornadas_trabajadas: LinearExpr = LinearExpr.sum(vars_por_trabajador[trabajador])
        jornadas_trabajadas_por_trabajador[trabajador] = total_jornadas_trabajadas

        if trabajador in set_voluntarios_doble:
            # Cada trabajador puede trabajar a lo sumo 2 jornadas, o 1 si no está en la lista_especialidades de
            # voluntarios para dobles.
            model.Add(total_jornadas_trabajadas <= 2)
            for jornada in jornadas: # type: Jornada
                # Cada trabajador solo puede desempeñar un puesto en cada jornada, no se pueden dividir en dos.
                model.Add(LinearExpr.sum(vars_por_trabajador_y_jornada[trabajador, jornada]) <= 1)
            # Se crea una variable nueva para representar si el trabajador en el que estamos actualmente iterando
            # realiza o no una jornada doble. Nótese que solo hacemos esto para los que son voluntarios para dobles.
            doble_jornada: IntVar = model.NewBoolVar(f'doble_jornada_{trabajador}')
            # Forzamos a que la variable doble_jornada sea verdadera cuando se trabajan 2 jornadas, y falsa en otro caso.
            model.Add(total_jornadas_trabajadas == 2).OnlyEnforceIf(doble_jornada)
            model.Add(total_jornadas_trabajadas != 2).OnlyEnforceIf(doble_jornada.Not())
            # Un trabajador que dobla jornadas solo puede hacerlo en las que está permitido (las de mañana y tarde)
            model.Add(LinearExpr.sum(vars_trabajador_no_doblar[trabajador]) == 0).OnlyEnforceIf(doble_jornada)
        else:
            # Si el trabajador no es voluntario para doble, solo podrá trabajar 1 jornada, sin más complicaciones.
            model.Add(total_jornadas_trabajadas <= 1)

    # Cada jornada debe cubrir su demanda.
    for puesto in puestos:
        for jornada in jornadas:
            model.Add(LinearExpr.sum(vars_por_puesto_y_jornada[puesto, jornada]) == demanda.get((puesto, jornada), 0))

    model.maximize(LinearExpr.sum([
        total_asignaciones_especialidades if maximizar_asignaciones_a_especialidad else 0,
        vol_noche_asignados_a_noche if maximizar_asignaciones_voluntarios_noche else 0,
        pref_manana_asignados_a_manana if maximizar_asignaciones_preferencias else 0,
        pref_tarde_asignados_a_tarde if maximizar_asignaciones_preferencias else 0,
    ]))

    solver: CpSolver = CpSolver()
    solver.parameters.num_search_workers = 8
    #solver.parameters.search_branching = cp_model.FIXED_SEARCH
    solver.parameters.max_time_in_seconds = 180.0
    solver.parameters.random_seed = 42
    solver.parameters.use_lns = True

    status: CpSolverStatus = solver.solve(model, SolutionHandler(vars))

    if verbose_general:
        print(f"Se encontraron {len(soluciones)} soluciones!")

    if verbose_estadisticas_avanzadas:
        print_estadisticas_avanzadas(solver, "Asignacion final: Estadísticas avanzadas")

    num_solucion: int = 0
    for resultado in soluciones:
        num_solucion += 1
        print(f"\nSolucion numero {num_solucion}:")

        numero_asignaciones_por_trabajador: dict[Trabajador, int] = defaultdict(int)
        voluntarios_noche_asignados_a_noche: int = 0
        preferencia_manana_asignados_a_manana: int = 0
        preferencia_tarde_asignados_a_tarde: int = 0
        for trabajador, puesto, jornada in resultado:  # type: Trabajador, PuestoTrabajo, Jornada
            numero_asignaciones_por_trabajador[trabajador] += 1
            if jornada in jornadas_noche and trabajador in set_voluntarios_noche:
                voluntarios_noche_asignados_a_noche += 1
            if jornada == Jornada.MANANA and trabajador in set_preferencia_manana:
                preferencia_manana_asignados_a_manana += 1
            if jornada == Jornada.TARDE and trabajador in set_preferencia_tarde:
                preferencia_tarde_asignados_a_tarde += 1

        trabajadores_asignados_dobles: set[Trabajador] = {
            trabajador
            for trabajador in trabajadores  # type: Trabajador
            if numero_asignaciones_por_trabajador[trabajador] == 2
        }

        trabajadores_asignados: int = len({trabajador for trabajador, _, _ in resultado})

        puestos_demandados: int = 0
        puestos_nocturnos_demandados: int = 0
        for (_, jornada), valor in demanda.items():  # type: tuple[PuestoTrabajo, Jornada], int
            if jornada in jornadas_noche:
                puestos_nocturnos_demandados += valor
            puestos_demandados += valor

        numero_voluntarios_noche: int = len(voluntarios_noche)
        numero_preferencia_manana: int = len(preferencia_manana)
        numero_preferencia_tarde: int = len(preferencia_tarde)
        asignaciones_especialidades_alcanzadas: int = solver.Value(total_asignaciones_especialidades)

        ultimo_codigo_asignado_por_especialidad: dict[PuestoTrabajo, int | None] = {
            puesto: max({trabajador.codigo for (trabajador, puesto, _) in resultado if
                         trabajador in sets_especialidades.get(puesto, {})}, default=None)
            for puesto in puestos  # type: PuestoTrabajo
        }

        trabajadores_voluntarios_asignados_noche: set[Trabajador] = {
            trabajador
            for trabajador, _, jornada in resultado  # type: Trabajador, PuestoTrabajo, Jornada
            if trabajador in set_voluntarios_noche and jornada in jornadas_noche
        }

        ultimo_codigo_voluntarios_noche: int | None = max({
            trabajador.codigo
            for trabajador in trabajadores_voluntarios_asignados_noche
        }, default=None)

        ultimo_codigo_voluntarios_doble: int | None = max({
            trabajador.codigo
            for trabajador in trabajadores_asignados_dobles  # type: Trabajador
        }, default=None)

        if verbose_general:

            print(f"\nSe demandaron {puestos_nocturnos_demandados} puestos nocturnos y hay {numero_voluntarios_noche} voluntarios de noche.")
            print(f"Se asignaron {voluntarios_noche_asignados_a_noche} de {numero_voluntarios_noche} ({100 * float(voluntarios_noche_asignados_a_noche) / numero_voluntarios_noche:.2f}%) voluntarios de noche a turnos de noche.\n")

            print(f"Se asignaron {preferencia_manana_asignados_a_manana} de {numero_preferencia_manana} ({100 * float(preferencia_manana_asignados_a_manana / numero_preferencia_manana):.2f}%) trabajadores con preferencia de mañana a turnos de mañana")
            print(f"Se asignaron {preferencia_tarde_asignados_a_tarde} de {numero_preferencia_tarde} ({100 * float(preferencia_tarde_asignados_a_tarde / numero_preferencia_tarde):.2f}%) trabajadores con preferencia de tarde a turnos de tarde")

            print(f'Número de asignaciones de especialidad alcanzado: {asignaciones_especialidades_alcanzadas} de {puestos_demandados} ({100 * asignaciones_especialidades_alcanzadas / puestos_demandados:.2f}%)')
            print(f'Número de trabajadores asignados: {trabajadores_asignados} de {num_trabajadores_disponibles} ({100 * float(trabajadores_asignados) / num_trabajadores_disponibles:.2f}%)')
            print(f'Puntuación alcanzada: {int(solver.ObjectiveValue())}\n')

        if verbose_asignacion_trabajadores:
            for trabajador, puesto, jornada in resultado:  # type: Trabajador, PuestoTrabajo, Jornada
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
            for jornada in jornadas:  # type: Jornada
                print(f'\n{jornada.nombre_es.capitalize()}:\n')

                for puesto in puestos:  # type: PuestoTrabajo
                    trabajadores_demandados: int = demanda.get((puesto, jornada), 0)
                    trabajadores_asignados_a_puesto_y_jornada: int = len(
                        {trabajador for trabajador, p, j in resultado if p == puesto and j == jornada})
                    if trabajadores_demandados != trabajadores_asignados_a_puesto_y_jornada:
                        print('**********************************************')
                        print(f'{puesto} en jornada {jornada}: MISMATCH!!!!')
                        print('**********************************************')
                    print(
                        f'{puesto.nombre_es} en jornada de {jornada.nombre_es} (demanda: {trabajadores_demandados}) asignado a {trabajadores_asignados_a_puesto_y_jornada} trabajadores:')

                    for trabajador in trabajadores:  # type: Trabajador
                        if (trabajador, puesto, jornada) in resultado:
                            preferencia: str = 'P.MAÑANA' if trabajador in set_preferencia_manana else 'P.TARDE' if trabajador in set_preferencia_tarde else ''
                            voluntario_noche: str = 'V.NOCHE' if trabajador in set_voluntarios_noche else ''
                            print(f'\tTrabajador {trabajador:<10}'
                                  f'\t{trabajador.capacidades[puesto].nombre_es:<10}'
                                  f'\t{preferencia:<10}'
                                  f'\t{voluntario_noche:<10}'
                                  )
                            for p, nivel in trabajador.capacidades.items():
                                print("\t\t", end="")
                                puesto_info: str = f'{p.nombre_es} ({nivel.nombre_es})'
                                print(f'{puesto_info:<15}', end="")
                            print(end="\n")


if __name__ == "__main__":
    realizar_asignacion(
        *data,
        verbose_estadisticas_avanzadas=True,
        verbose_general=True,
        verbose_asignacion_trabajadores=False,
        verbose_asignacion_puestos=False,
        maximizar_asignaciones_a_especialidad=True,
        maximizar_asignaciones_voluntarios_noche=True,
        maximizar_asignaciones_preferencias=True
    )