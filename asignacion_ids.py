from __future__ import annotations

from collections import defaultdict

from ortools.sat.cp_model_pb2 import CpSolverStatus
from ortools.sat.python import cp_model
from ortools.sat.python.cp_model import CpModel, IntVar, LinearExpr, CpSolver

from ClasesMetodosAuxiliares import ListasPreferencias, ParametrosPuntuacion, Verbose, DatosTrabajadoresPuestosJornadas, \
    Asignacion, print_estadisticas_avanzadas, formatear_float, IdsListasPreferencias, IdsParametrosPuntuacion, \
    IdsTrabajadoresPuestosJornadas, IdsAsignacion
from Clases import Trabajador, PuestoTrabajo, Jornada, TipoJornada, NivelDesempeno
from parse_ids import data_ids


def calcular_coeficientes_puntuacion(
    trabajadores: list[int],
    jornadas: set[int],
    disponibilidad: set[tuple[int, int]], # (trabajador_id, jornada_id)
    listas_preferencias: IdsListasPreferencias,
    parametros: IdsParametrosPuntuacion
) -> tuple[
    dict[tuple[int, int, int], int], # (trabajador, puesto, jornada) -> puntuación por realizar esta asignación
    dict[int, int] # trabajador -> puntuación por asignar a este trabajador a dobles
]:
    coeficientes_asignaciones: dict[tuple[int, int, int], int] = {} # (trabajador_id, puesto_id, jornada_id) -> coeficiente para esa asignación
    coeficientes_dobles: dict[int, int] = {} # trabajador -> puntuación por asignar ese trabajador a dobles

    especialidades, preferencia_por_jornada, voluntarios_doble = listas_preferencias

    (
        max_especialidad, decay_especialidad,
        max_capacidad, decay_capacidad,
        max_voluntarios_doble, decay_voluntarios_doble,
        max_preferencia_por_jornada, decay_preferencia_por_jornada, penalizacion_por_jornada
    ) = parametros.unpack()

    for trabajador_id in voluntarios_doble:
        coeficientes_dobles[trabajador_id] = max_voluntarios_doble - decay_voluntarios_doble * voluntarios_doble.index(trabajador_id)

    for trabajador_id in trabajadores:
        trabajador_capacidades_id = Trabajador.from_id(trabajador_id).capacidades_ids
        for puesto_id in trabajador_capacidades_id:
            for jornada_id in jornadas:
                if (trabajador_id, jornada_id) in disponibilidad:
                    # Puntuación por capacidad.
                    puntuacion_capacidad: int = max(0, max_capacidad - decay_capacidad * (trabajador_capacidades_id[puesto_id] - 1))

                    # Puntuación por estar más alto en las listas de especialidades.
                    puntuacion_especialidad: int = 0
                    especialistas_puesto: list[int] = especialidades.get(puesto_id, set())
                    if trabajador_id in especialistas_puesto:
                        puntuacion_especialidad = max(0, max_especialidad - decay_especialidad * especialistas_puesto.index(trabajador_id))

                    # Puntuación por preferencia de jornada o penalización por no ser voluntario para noche
                    puntuacion_jornada: int = 0
                    if jornada_id in Jornada.jornadas_con_preferencia():
                        tipo_jornada_id = Jornada.from_id(jornada_id).tipo_jornada_id
                        if trabajador_id in preferencia_por_jornada[tipo_jornada_id]:
                            puntuacion_jornada += max_preferencia_por_jornada[tipo_jornada_id] - decay_preferencia_por_jornada[tipo_jornada_id] * preferencia_por_jornada[tipo_jornada_id].index(trabajador_id)
                        else:
                            puntuacion_jornada -= penalizacion_por_jornada[tipo_jornada_id]

                    coeficientes_asignaciones[trabajador_id, puesto_id, jornada_id] = puntuacion_capacidad + puntuacion_especialidad + puntuacion_jornada

    return coeficientes_asignaciones, coeficientes_dobles


def realizar_asignacion(
    # Datos básicos: listas de los IDs de los trabajadores, puestos y jornadas.
    datos: IdsTrabajadoresPuestosJornadas,

    # Listas de preferencias a respetar: una por especialidad, por tipo de jornada y de voluntarios a coeficientes_dobles.
    listas_preferencias: IdsListasPreferencias,

    # (puesto, jornada) -> demanda de trabajadores para ese puesto en esa jornada.
    demanda: dict[tuple[int, int], int],

    # Tuplas (trabajador, jornada) en las que el trabajador está disponible en esa jornada.
    disponibilidad: set[tuple[int, int]],

    # Parámetros para controlar si se imprime por pantalla la solución encontrada y estadísticas sobre la resolución.
    verbose: Verbose = Verbose(False, False, False, False),

    # Parámetros para el cálculo de la función objetivo a maximizar
    parametros: IdsParametrosPuntuacion = IdsParametrosPuntuacion()
) -> set[tuple[int, int, int]]:
    """
    Retorna el resultado como un conjunto de tuplas (trabajador_id, puesto_id, jornada_id) representando que ese
    trabajador fue asignado a ese puesto en esa jornada.
    """

    # ******************************************************************************************************************
    # *********************************** DESEMPAQUETADO Y PRECOMPUTACIÓN **********************************************
    # ******************************************************************************************************************

    # Se desempaquetan los datos básicos y las listas de preferencias.
    trabajadores, puestos, jornadas = datos
    especialidades, preferencias_jornada, voluntarios_doble = listas_preferencias

    # Se precomputan varios sets para comprobaciones de membresía más rápidas.
    # No se pueden recibir los datos como sets directamente porque el orden es importante en la asignación.
    set_voluntarios_doble: set[int] = set(voluntarios_doble)
    set_preferencias_por_jornada: dict[int, set[int]] = {
        tipo_jornada : set(lista_trabajadores)
        for tipo_jornada, lista_trabajadores in preferencias_jornada.items()
    }
    sets_especialidades: dict[int, set[int]] = {
        puesto : set(lista_trabajadores)
        for puesto, lista_trabajadores in especialidades.items()
    }

    # A partir de una función auxiliar, se calculan los coeficientes de puntuación de cada asignación y de asignar o no
    # a un trabajador a doble jornada. Además, se calcula implícitamente en las llaves del diccionario
    # coeficientes_asignaciones las combinaciones de (trabajador, puesto, jornada) válidas.
    coeficientes_asignaciones, coeficientes_dobles = calcular_coeficientes_puntuacion(
        trabajadores,
        jornadas,
        disponibilidad,
        listas_preferencias,
        parametros
    )

    # ******************************************************************************************************************
    # *********************************** CREACIÓN DEL MODELO Y VARIABLES **********************************************
    # ******************************************************************************************************************

    model: CpModel = CpModel()

    # Se guardarán tuplas Asignacion(trabajador, puesto, jornada, var, puntuacion), a partir de las cuales se accederá
    # a las variables del modelo. La variable de una asignacion representa la decisión binaria de realizar tal
    # asignación o no.
    asignaciones: list[IdsAsignacion] = []

    # Por la lógica de calcular_coeficientes_puntuacion con la que se construyó el diccionario coeficientes_asignaciones,
    # en el siguiente bucle for solo se iterará en las combinaciones (trabajador, puesto, jornada) válidas, es decir,
    # en las que el trabajador sea capaz de desempeñar el puesto y esté disponible en esa jornada.
    for trabajador_id, puesto_id, jornada_id in coeficientes_asignaciones:
        asignaciones.append(IdsAsignacion(
            trabajador_id,
            puesto_id,
            jornada_id,
            var=model.NewBoolVar(f'x_{trabajador_id}_{puesto_id}_{jornada_id}'),
            puntuacion=coeficientes_asignaciones[trabajador_id, puesto_id, jornada_id]
        ))

    # Se guardan expresiones lineales obtenidas al sumar las variables de las asignaciones que verifican ciertas
    # condiciones de interés: asignar un trabajador a su especialidad y respetar una preferencia de jornada.
    total_asignaciones_especialidades: LinearExpr = LinearExpr.Sum([
        asignacion.var
        for asignacion in asignaciones
        if asignacion.puesto_id in asignacion.get_trabajador().especialidades_ids
    ])

    total_asignaciones_respeta_jornada: dict[int, LinearExpr] = {
        tipo_jornada_id : LinearExpr.Sum([
            asignacion.var
            for asignacion in asignaciones
            if asignacion.jornada_id in Jornada.jornadas_con_preferencia()
               and asignacion.get_jornada().tipo_jornada_id == tipo_jornada_id
               and asignacion.trabajador_id in set_preferencias_por_jornada[tipo_jornada_id]
        ])
        for tipo_jornada_id in TipoJornada.get_registro().keys()
    }

    # ******************************************************************************************************************
    # ************************************** RESTRICCIONES DEL MODELO **************************************************
    # ******************************************************************************************************************

    # Se preparan diccionarios para almacenar expresiones lineales y variables que representan el total de jornadas
    # trabajadas por cada trabajador y si un trabajador de la lista de voluntarios a dobles realiza una doble jornada.
    jornadas_trabajadas_por_trabajador: dict[int, LinearExpr] = {}
    dobles_por_trabajador: dict[int, IntVar] = {}

    for trabajador_id in trabajadores:
        # Se crea una expresión lineal que representa el total de jornadas trabajadas por el trabajador
        trabajador_capacidades: dict[int, int] = Trabajador.from_id(trabajador_id).capacidades_ids
        total_jornadas_trabajadas: LinearExpr = LinearExpr.Sum([
            asignacion.var
            for asignacion in asignaciones
            if asignacion.trabajador_id == trabajador_id
        ])
        jornadas_trabajadas_por_trabajador[trabajador_id] = total_jornadas_trabajadas

        if trabajador_id in set_voluntarios_doble:
            # Cada trabajador voluntario para dobles puede trabajar a lo sumo 2 jornadas.
            model.Add(total_jornadas_trabajadas <= 2)

            for jornada_id in jornadas:
                # Cada trabajador solo puede desempeñar un puesto en cada jornada, no se puede dividir en dos.
                model.Add(LinearExpr.Sum([
                    asignacion.var
                    for asignacion in asignaciones
                    if asignacion.trabajador_id == trabajador_id
                       and asignacion.jornada_id == jornada_id
                ]) <= 1)

            # Se crea una variable nueva para representar si el trabajador en el que estamos actualmente iterando
            # realiza o no una jornada doble. Nótese que se hace esto para los que son voluntarios para dobles.
            doble_jornada: IntVar = model.NewBoolVar(f'doble_jornada_{trabajador_id}')
            # Forzamos a que la variable doble_jornada sea verdadera cuando se trabajan 2 jornadas, y falsa en otro caso.
            model.Add(total_jornadas_trabajadas == 2).OnlyEnforceIf(doble_jornada)
            model.Add(total_jornadas_trabajadas != 2).OnlyEnforceIf(doble_jornada.Not())
            # Se guarda la variable en un diccionario previamente creado para su posterior acceso.
            dobles_por_trabajador[trabajador_id] = doble_jornada
            # Un trabajador que dobla jornadas solo puede hacerlo en las que está permitido (las de mañana y tarde)
            # Luego si doble_jornada es cierto, trabajará 0 turnos en las jornadas en las que no se puede doblar.
            model.Add(LinearExpr.Sum([
                asignacion.var
                for asignacion in asignaciones
                if asignacion.trabajador_id == trabajador_id
                   and not asignacion.get_jornada().puede_doblar
            ]) == 0).OnlyEnforceIf(doble_jornada)

        else:
            # Si el trabajador no es voluntario para doble, solo podrá trabajar 1 jornada, sin más complicaciones.
            model.Add(total_jornadas_trabajadas <= 1)

    # Cada jornada debe cubrir su demanda.
    for puesto_id in puestos:
        for jornada_id in jornadas:
            model.Add(LinearExpr.Sum([
                asignacion.var
                for asignacion in asignaciones
                if asignacion.puesto_id == puesto_id
                   and asignacion.jornada_id == jornada_id
            ]) == demanda.get((puesto_id, jornada_id), 0))

    # ******************************************************************************************************************
    # ************************************* FUNCIÓN OBJETIVO A MAXIMIZAR ***********************************************
    # ******************************************************************************************************************

    puntuacion_asignaciones: LinearExpr = LinearExpr.Sum([
        asignacion.puntuacion * asignacion.var
        for asignacion in asignaciones
        if asignacion.puntuacion != 0
    ])

    puntuacion_dobles: LinearExpr = LinearExpr.Sum([
        coeficientes_dobles[trabajador_id] * dobles_por_trabajador[trabajador_id]
        for trabajador_id in dobles_por_trabajador
        if coeficientes_dobles[trabajador_id] != 0
    ])

    # Se maximiza la puntuación obtenida por las asignaciones más las asignaciones a coeficientes_dobles.
    model.Maximize(LinearExpr.Sum(puntuacion_asignaciones, puntuacion_dobles))

    # ******************************************************************************************************************
    # *********************************************** RESOLUCIÓN *******************************************************
    # ******************************************************************************************************************

    solver: CpSolver = CpSolver()
    solver.parameters.num_search_workers = 8
    solver.parameters.search_branching = cp_model.FIXED_SEARCH
    #solver.parameters.linearization_level = 0
    solver.parameters.random_seed = 10
    #solver.parameters.use_lns = True

    status: CpSolverStatus = solver.Solve(model)

    # ******************************************************************************************************************
    # *********************************** CÁLCULO Y VISUALIZACIÓN DEL RESULTADO ****************************************
    # ******************************************************************************************************************

    if verbose.general:
        if status == cp_model.OPTIMAL:
            print("Se encontró una solución óptima.")
        elif status == cp_model.FEASIBLE:
            print("Se encontró una solución factible, pero no necesariamente óptima.")
        else:
            print(f"No se encontraron soluciones.")

    if verbose.estadisticas_avanzadas:
        print_estadisticas_avanzadas(solver, "Estadísticas avanzadas")

    resultado: set[tuple[int, int, int]] = {
        (asignacion.trabajador_id, asignacion.puesto_id, asignacion.jornada_id)
        for asignacion in asignaciones
        if solver.Value(asignacion.var) == 1
    }

    trabajadores_asignados_dobles: set[int] = {
        trabajador_id
        for trabajador_id, _, _ in resultado
        if solver.Value(jornadas_trabajadas_por_trabajador[trabajador_id]) == 2
    }

    ultimo_codigo_asignado_por_especialidad: dict[int, int | None] = {
        puesto_id : max({
            Trabajador.from_id(trabajador_id).codigo
            for (trabajador_id, puesto_id_, _) in resultado
            if puesto_id == puesto_id_
               and trabajador_id in sets_especialidades.get(puesto_id, {})
        }, default=None)
        for puesto_id in puestos
    }

    ultimo_codigo_asignado_por_jornada: dict[int, int | None] = {
        tipo_jornada_id : max({
            Trabajador.from_id(trabajador_id).codigo
            for (trabajador_id, _, jornada_id) in resultado
            if jornada_id in Jornada.jornadas_con_preferencia()
                and Jornada.from_id(jornada_id).tipo_jornada_id == tipo_jornada_id
                and trabajador_id in set_preferencias_por_jornada[tipo_jornada_id]
        }, default=None)
        for tipo_jornada_id in TipoJornada.get_registro()
    }

    ultimo_codigo_voluntarios_doble: int | None = max({
        Trabajador.from_id(trabajador_id).codigo
        for trabajador_id in trabajadores_asignados_dobles
    }, default=None)


    if verbose.general:

        num_trabajadores_asignados: int = len({trabajador_id for trabajador_id, _, _ in resultado})
        num_trabajadores_disponibles: int = len({asignacion.trabajador_id for asignacion in asignaciones})

        num_preferencia_manana: int = len({
            asignacion.trabajador_id
            for asignacion in asignaciones
            if asignacion.trabajador_id in set_preferencias_por_jornada[1]
        })

        num_preferencia_tarde: int = len({
            asignacion.trabajador_id
            for asignacion in asignaciones
            if asignacion.trabajador_id in set_preferencias_por_jornada[2]
        })

        num_voluntarios_noche: int = len({
            asignacion.trabajador_id
            for asignacion in asignaciones
            if asignacion.trabajador_id in set_preferencias_por_jornada[3]
        })

        num_voluntarios_dobles: int = len({
            asignacion.trabajador_id
            for asignacion in asignaciones
            if asignacion.trabajador_id in set_voluntarios_doble
        })

        puestos_demandados: int = 0
        puestos_demandados_por_jornada: dict[int, int] = defaultdict(lambda: 0)
        for (_, jornada_id), valor in demanda.items():
            puestos_demandados += valor
            if jornada_id in Jornada.jornadas_con_preferencia():
                puestos_demandados_por_jornada[Jornada.from_id(jornada_id).tipo_jornada] += valor

        puestos_demandados_manana: int = puestos_demandados_por_jornada[TipoJornada.manana_Bilbao_id]
        puestos_demandados_tarde: int = puestos_demandados_por_jornada[TipoJornada.tarde_Bilbao_id]
        puestos_demandados_noche: int = puestos_demandados_por_jornada[TipoJornada.noche_Bilbao_id]

        num_asignaciones_especialidad: int = solver.Value(total_asignaciones_especialidades)
        num_pref_manana_asignados_manana: int = solver.Value(total_asignaciones_respeta_jornada[TipoJornada.manana_Bilbao_id])
        num_pref_tarde_asignados_tarde: int = solver.Value(total_asignaciones_respeta_jornada[TipoJornada.tarde_Bilbao_id])
        num_vol_noche_asignados_noche: int = solver.Value(total_asignaciones_respeta_jornada[TipoJornada.noche_Bilbao_id])

        print(f"{'Turno':<10} | {'Puestos demandados':<20} | {'Preferencia/voluntarios'}")
        print(f"{'Mañana':<10} | {puestos_demandados_manana:<20} | {num_preferencia_manana}")
        print(f"{'Tarde':<10} | {puestos_demandados_tarde:<20} | {num_preferencia_tarde}")
        print(f"{'Noche':<10} | {puestos_demandados_noche:<20} | {num_voluntarios_noche}")
        print("-" * 60)
        print(f"{'Dobles':<10} | {'*':<20} | {num_voluntarios_dobles}")

        print(end="\n")
        print(f"Se asignaron {num_pref_manana_asignados_manana} de {num_preferencia_manana} ({formatear_float(100 * float(num_pref_manana_asignados_manana / num_preferencia_manana))}%) trabajadores con preferencia de mañana a turnos de mañana.")
        print(f"Se cubrieron {num_pref_manana_asignados_manana} de {puestos_demandados_manana} ({formatear_float(100 * float(num_pref_manana_asignados_manana) / puestos_demandados_manana)}%) puestos de mañana con trabajadores con preferencia de mañana.")

        print(end="\n")
        print(f"Se asignaron {num_pref_tarde_asignados_tarde} de {num_preferencia_tarde} ({formatear_float(100 * float(num_pref_tarde_asignados_tarde) / num_preferencia_tarde)}%) trabajadores con preferencia de tarde a turnos de tarde.")
        print(f"Se cubrieron {num_pref_tarde_asignados_tarde} de {puestos_demandados_tarde} ({formatear_float(100 * float(num_pref_tarde_asignados_tarde) / puestos_demandados_tarde)}%) puestos de tarde con trabajadores con preferencia de tarde.")

        print(end="\n")
        print(f"Se asignaron {num_vol_noche_asignados_noche} de {num_voluntarios_noche} ({formatear_float(100 * float(num_vol_noche_asignados_noche) / num_voluntarios_noche)}%) voluntarios de noche a turnos de noche.")
        print(f"Se cubrieron {num_vol_noche_asignados_noche} de {puestos_demandados_noche} ({formatear_float(100 * float(num_vol_noche_asignados_noche) / puestos_demandados_noche)}%) puestos de noche con voluntarios de noche.")

        print(end="\n")
        print(f'Número de asignaciones de especialidad alcanzado: {num_asignaciones_especialidad} de {puestos_demandados} ({formatear_float(100 * float(num_asignaciones_especialidad) / puestos_demandados)}%)')
        print(f'Número de trabajadores asignados: {num_trabajadores_asignados} de {num_trabajadores_disponibles} ({formatear_float(100 * float(num_trabajadores_asignados) / num_trabajadores_disponibles)}%)')

        print(end="\n")
        print(f'Puntuación alcanzada: {int(solver.ObjectiveValue())}\n')


    if verbose.asignacion_trabajadores:
        for trabajador_id, puesto_id, jornada_id in resultado:
            realiza_doble: str = 'DOBLE' if trabajador_id in trabajadores_asignados_dobles else ''
            polivalencia: str = Trabajador.from_id(trabajador_id).capacidades_ids[puesto_id].nombre_es + (" ("+str(especialidades.get(puesto_id, []).index(trabajador_id))+")" if puesto_id in trabajador_id.especialidades and puesto_id in especialidades else "")
            preferencia: str = 'P.MAÑANA' + f' ({preferencias_jornada[TipoJornada.manana_Bilbao_id].index(trabajador_id)})' if trabajador_id in set_preferencias_por_jornada[TipoJornada.manana_Bilbao_id] else 'P.TARDE' + f' ({preferencias_jornada[TipoJornada.tarde_Bilbao_id].index(trabajador_id)})' if trabajador_id in set_preferencias_por_jornada[TipoJornada.tarde_Bilbao_id] else ''
            voluntario_noche: str = 'V.NOCHE' + f' ({preferencias_jornada[TipoJornada.noche_Bilbao_id].index(trabajador_id)})' if trabajador_id in set_preferencias_por_jornada[TipoJornada.noche_Bilbao_id] else ''
            voluntario_doble: str = 'V.DOBLE' + f' ({voluntarios_doble.index(trabajador_id)})' if trabajador_id in set_voluntarios_doble else ''
            puntuacion_por_asignacion: str = 'Punt: ' + str(coeficientes_asignaciones[trabajador_id, puesto_id, jornada_id] + solver.Value(dobles_por_trabajador.get(trabajador_id, 0)) * coeficientes_dobles.get(trabajador_id, 0))

            print(
                f"Trabajador {trabajador_id:<{3}} -> "
                f"{puesto_id.nombre_es:<{21}} | "
                f"{jornada_id.nombre_es:<{10}}"
                f"{realiza_doble:<{10}}{polivalencia:<{30}}"
                f"{preferencia:<{15}}"
                f"{voluntario_noche:<{15}}"
                f"{voluntario_doble:<{15}}"
                f"{puntuacion_por_asignacion:<{7}}"
            )


    if verbose.asignacion_puestos:
        if verbose.asignacion_trabajadores:
            print("\n\n")

        for puesto_id in puestos:
            print(f'{puesto_id.nombre_es}:')
            for jornada_id in jornadas:
                trabajadores_demandados: int = demanda.get((puesto_id, jornada_id), 0)
                trabajadores_asignados_a_puesto_y_jornada: int = len(
                    {trabajador for trabajador, p, j in resultado if p == puesto_id and j == jornada_id})
                if trabajadores_demandados != trabajadores_asignados_a_puesto_y_jornada:
                    print('**********************************************')
                    print(f'{puesto_id} en jornada {jornada_id}: MISMATCH!!!!')
                    print('**********************************************')
                if trabajadores_demandados != 0:
                    print('\t' f'{Jornada.from_id(jornada_id).nombre_es} (demanda: {trabajadores_demandados}) asignado a {trabajadores_asignados_a_puesto_y_jornada} trabajadores:')

                    for trabajador_id in trabajadores:
                        if (trabajador_id, puesto_id, jornada_id) in resultado:
                            preferencia: str = 'P.MAÑANA' if trabajador_id in set_preferencias_por_jornada[TipoJornada.manana_Bilbao_id] else 'P.TARDE' if trabajador_id in set_preferencias_por_jornada[TipoJornada.tarde_Bilbao_id] else ''
                            voluntario_noche: str = 'V.NOCHE' if trabajador_id in set_preferencias_por_jornada[TipoJornada.noche_Bilbao_id] else ''
                            info_puesto: str = f'{puesto_id.nombre_es} | {Trabajador.from_id(trabajador_id).capacidades_ids[puesto_id].nombre_es}' + (f' ({especialidades[puesto_id].index(trabajador_id)})' if trabajador_id in sets_especialidades[puesto_id] else '')
                            print(
                                f'\t\tTrabajador {trabajador_id:<{5}}'
                                f' -> '
                                f'{info_puesto:<{40}}'
                                f' {preferencia:<{15}}'
                                f'{voluntario_noche:<{15}}'
                            )

                            for p in Trabajador.from_id(trabajador_id).especialidades_ids - {puesto_id}:
                                print("\t\t\t", end="")
                                puesto_info: str = f'{p.nombre_es}: {especialidades[p].index(trabajador_id)}'
                                print(f'{puesto_info:<15}')

    return resultado, solver


def comparar_asignaciones(
    asignacion1: set[tuple[Trabajador, PuestoTrabajo, Jornada]],
    asignacion2: set[tuple[Trabajador, PuestoTrabajo, Jornada]],
) -> None:
    trabajadores_que_no_coinciden: set[Trabajador] = {
        trabajador
        for (trabajador, _, _) in asignacion1 ^ asignacion2 # Diferencia simétrica de conjuntos
    }
    dict_asignacion1: dict[Trabajador, tuple[PuestoTrabajo, Jornada]] = {
        trabajador : (puesto, jornada)
        for (trabajador, puesto, jornada) in asignacion1
    }
    dict_asignacion2: dict[Trabajador, tuple[PuestoTrabajo, Jornada]] = {
        trabajador : (puesto, jornada)
        for (trabajador, puesto, jornada) in asignacion2
    }
    print(f"Hay {len(trabajadores_que_no_coinciden)} trabajadores que no coinciden.")
    print(f"{'Solo en asignación 1':<45} | {'Solo en asignación 2':<45}")
    print('-' * 85)
    for trabajador in trabajadores_que_no_coinciden:
        col1: str = f"{trabajador}, {dict_asignacion1[trabajador][0].nombre_es}, {dict_asignacion1[trabajador][1].nombre_es}" if trabajador in dict_asignacion1 else f"{trabajador.codigo} no asignado."
        col2: str = f"{trabajador}, {dict_asignacion2[trabajador][0].nombre_es}, {dict_asignacion2[trabajador][1].nombre_es}" if trabajador in dict_asignacion2 else f"{trabajador.codigo} no asignado."
        print(f"{col1:<45} | {col2:<45}")


def test_tiempo_ejecucion(n: int, datos) -> None:
    print("Test de eficiencia:")
    tiempo_total: int = 0
    for i in range(n):
        _, solver = realizar_asignacion(
            *datos,
            verbose=Verbose(
                general=False,
                estadisticas_avanzadas=False,
                asignacion_puestos=False,
                asignacion_trabajadores=False
            )
        )
        tiempo = solver.wall_time
        tiempo_total += tiempo
        print(f"Iteración {i + 1:<2}: {tiempo}")
    tiempo_total: float | None = tiempo_total / n if tiempo_total != 0 else None
    if tiempo_total:
        print(f"Tiempo medio de ejecución: {tiempo_total}")
    else:
        print("Error, el tiempo total guardado es nulo.")

if __name__ == "__main__":
    datos = data_ids
    resultado = realizar_asignacion(
        *datos,
        verbose=Verbose(
            general=True,
            estadisticas_avanzadas=True,
            asignacion_puestos=False,
            asignacion_trabajadores=True
        )
    )