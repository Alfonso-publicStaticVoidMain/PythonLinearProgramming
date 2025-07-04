from __future__ import annotations

from collections import defaultdict

from ortools.sat.cp_model_pb2 import CpSolverStatus
from ortools.sat.python import cp_model
from ortools.sat.python.cp_model import CpModel, IntVar, LinearExpr, CpSolver

from ClasesAuxiliares import ListasPreferencias, ParametrosPuntuacion, Verbose, DatosTrabajadoresPuestosJornadas, Asignacion
from Clases import Trabajador, PuestoTrabajo, Jornada, TipoJornada, NivelDesempeno
from parse import data


def formatear_float(
    valor: float,
    *,
    posiciones_decimales: int = 2,
    tolerancia: float = 1e-3
) -> str:
    """
    Devuelve una cadena que representa el número en coma flotante recibido como argumento, redondeando al entero más
    cercano si está a una distancia de él de a lo sumo la tolerancia especificada, o representándolo como número decimal
    con la cantidad deseada de posiciones decimales en otro caso.
    """
    entero_mas_cercano: int = round(valor)
    if abs(valor - entero_mas_cercano) < tolerancia:
        return str(entero_mas_cercano)
    return f"{valor:.{posiciones_decimales}f}"


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


def calcular_coeficientes_puntuacion(
    trabajadores: list[Trabajador],
    jornadas: set[Jornada],
    disponibilidad: set[tuple[Trabajador, Jornada]],
    listas_preferencias: ListasPreferencias,
    parametros: ParametrosPuntuacion
) -> tuple[
    dict[tuple[Trabajador, PuestoTrabajo, Jornada], int], # (trabajador, puesto, jornada) -> puntuación por realizar esta asignación
    dict[Trabajador, int] # trabajador -> puntuación por asignar a este trabajador a dobles
]:
    coeficientes_asignaciones: dict[tuple[Trabajador, PuestoTrabajo, Jornada], int] = {}
    coeficientes_dobles: dict[Trabajador, int] = {}

    especialidades, preferencia_por_jornada, voluntarios_doble = listas_preferencias

    (
        max_especialidad, decay_especialidad,
        max_capacidad, decay_capacidad,
        max_voluntarios_doble, decay_voluntarios_doble,
        max_preferencia_por_jornada, decay_preferencia_por_jornada, penalizacion_por_jornada
    ) = parametros.unpack()

    for trabajador in voluntarios_doble:
        coeficientes_dobles[trabajador] = max_voluntarios_doble - decay_voluntarios_doble * voluntarios_doble.index(trabajador)

    for trabajador in trabajadores:
        for puesto in trabajador.capacidades:
            for jornada in jornadas:
                if (trabajador, jornada) in disponibilidad:
                    # Puntuación por capacidad.
                    puntuacion_capacidad: int = max(0, max_capacidad - decay_capacidad * (trabajador.capacidades[puesto].id - 1))

                    # Puntuación por estar más alto en las listas de especialidades.
                    puntuacion_especialidad: int = 0
                    especialistas_puesto: list[Trabajador] = especialidades.get(puesto, [])
                    if trabajador in especialistas_puesto:
                        puntuacion_especialidad = max(0, max_especialidad - decay_especialidad * especialistas_puesto.index(trabajador))

                    # Puntuación por preferencia de jornada o penalización por no ser voluntario para noche
                    puntuacion_jornada: int = 0
                    if jornada in Jornada.jornadas_con_preferencia():
                        tipo_jornada = jornada.tipo_jornada
                        if trabajador in preferencia_por_jornada[tipo_jornada]:
                            puntuacion_jornada += max_preferencia_por_jornada[tipo_jornada] - decay_preferencia_por_jornada[tipo_jornada] * preferencia_por_jornada[tipo_jornada].index(trabajador)
                        else:
                            puntuacion_jornada -= penalizacion_por_jornada[tipo_jornada]

                    coeficientes_asignaciones[trabajador, puesto, jornada] = puntuacion_capacidad + puntuacion_especialidad + puntuacion_jornada

    return coeficientes_asignaciones, coeficientes_dobles


def realizar_asignacion(
    # Datos básicos: listas de trabajadores, puestos y jornadas.
    datos: DatosTrabajadoresPuestosJornadas,
    # Listas de preferencias a respetar: una por especialidad, por tipo de jornada y de voluntarios a coeficientes_dobles.
    listas_preferencias: ListasPreferencias,
    # (puesto, jornada) -> demanda de trabajadores para ese puesto en esa jornada.
    demanda: dict[tuple[PuestoTrabajo, Jornada], int],
    # Tuplas (trabajador, jornada) en las que el trabajador está disponible en esa jornada.
    disponibilidad: set[tuple[Trabajador, Jornada]],
    # Parámetros para controlar si se imprime por pantalla la solución encontrada y estadísticas sobre la resolución.
    verbose: Verbose = Verbose(False, False, False, False),
    # Parámetros para el cálculo de la función objetivo a maximizar
    parametros: ParametrosPuntuacion = ParametrosPuntuacion()
) -> set[tuple[Trabajador, PuestoTrabajo, Jornada]]:

    # ******************************************************************************************************************
    # *********************************** DESEMPAQUETADO Y PRECOMPUTACIÓN **********************************************
    # ******************************************************************************************************************

    # Se desempaquetan los datos básicos y las listas de preferencias.
    trabajadores, puestos, jornadas = datos
    especialidades, preferencias_jornada, voluntarios_doble = listas_preferencias

    # Se precomputan varios sets para comprobaciones de membresía más rápidas.
    # No se pueden recibir los datos como sets directamente porque el orden es importante en la asignación.
    set_voluntarios_doble: set[Trabajador] = set(voluntarios_doble)
    set_preferencias_por_jornada: dict[TipoJornada, set[Trabajador]] = {
        tipo_jornada : set(lista_trabajadores)
        for tipo_jornada, lista_trabajadores in preferencias_jornada.items()
    }
    sets_especialidades: dict[PuestoTrabajo, set[Trabajador]] = {
        puesto : set(lista_trabajadores)
        for puesto, lista_trabajadores in especialidades.items()
    }

    # A partir de una función auxiliar, se calculan los coeficientes de puntuación de cada asignación y de asignar o no
    # a un trabajador a doble jornada. Además, se calcula implícitamente en las llaves del diccionario
    # coeficientes_asignaciones las combinaciones de (trabajador, puesto, jornada) válidas.
    coeficientes_asignaciones: dict[tuple[Trabajador, PuestoTrabajo, Jornada], int]
    coeficientes_dobles: dict[Trabajador, int]
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
    asignaciones: list[Asignacion] = []

    # Por la lógica de calcular_coeficientes_puntuacion con la que se construyó el diccionario coeficientes_asignaciones,
    # en el siguiente bucle for solo se iterará en las combinaciones (trabajador, puesto, jornada) válidas, es decir,
    # en las que el trabajador sea capaz de desempeñar el puesto y esté disponible en esa jornada.
    for trabajador, puesto, jornada in coeficientes_asignaciones:
        asignaciones.append(Asignacion(
            trabajador,
            puesto,
            jornada,
            var=model.NewBoolVar(f'x_{trabajador}_{puesto}_{jornada}'),
            puntuacion=coeficientes_asignaciones[trabajador, puesto, jornada]
        ))

    # Se guardan expresiones lineales obtenidas al sumar las variables de las asignaciones que verifican ciertas
    # condiciones de interés: asignar un trabajador a su especialidad y respetar una preferencia de jornada.
    total_asignaciones_especialidades: LinearExpr = LinearExpr.Sum([
        asignacion.var
        for asignacion in asignaciones
        if asignacion.puesto in asignacion.trabajador.especialidades
    ])
    total_asignaciones_respeta_jornada: dict[TipoJornada, LinearExpr] = {
        tipo_jornada : LinearExpr.Sum([
            asignacion.var
            for asignacion in asignaciones
            if asignacion.jornada in Jornada.jornadas_con_preferencia()
            and asignacion.jornada.tipo_jornada == tipo_jornada
            and asignacion.trabajador in set_preferencias_por_jornada[tipo_jornada]
        ])
        for tipo_jornada in TipoJornada
    }

    # Se preparan diccionarios para almacenar expresiones lineales y variables que representan el total de jornadas
    # trabajadas por cada trabajador y si un trabajador de la lista de voluntarios a dobles realiza una doble jornada.
    jornadas_trabajadas_por_trabajador: dict[Trabajador, LinearExpr] = {}
    dobles_por_trabajador: dict[Trabajador, IntVar] = {}

    # ******************************************************************************************************************
    # ************************************** RESTRICCIONES DEL MODELO **************************************************
    # ******************************************************************************************************************

    for trabajador in trabajadores:
        # Se crea una expresión lineal que representa el total de jornadas trabajadas por el trabajador
        trabajador_capacidades: dict[PuestoTrabajo, NivelDesempeno] = trabajador.capacidades
        total_jornadas_trabajadas: LinearExpr = LinearExpr.Sum([
            asignacion.var
            for asignacion in asignaciones
            if asignacion.trabajador == trabajador
        ])
        jornadas_trabajadas_por_trabajador[trabajador] = total_jornadas_trabajadas

        if trabajador in set_voluntarios_doble:
            # Cada trabajador voluntario para dobles puede trabajar a lo sumo 2 jornadas.
            model.Add(total_jornadas_trabajadas <= 2)

            for jornada in jornadas:
                # Cada trabajador solo puede desempeñar un puesto en cada jornada, no se puede dividir en dos.
                model.Add(LinearExpr.Sum([
                    asignacion.var
                    for asignacion in asignaciones
                    if asignacion.trabajador == trabajador
                    and asignacion.jornada == jornada
                ]) <= 1)

            # Se crea una variable nueva para representar si el trabajador en el que estamos actualmente iterando
            # realiza o no una jornada doble. Nótese que se hace esto para los que son voluntarios para dobles.
            doble_jornada: IntVar = model.NewBoolVar(f'doble_jornada_{trabajador}')
            # Forzamos a que la variable doble_jornada sea verdadera cuando se trabajan 2 jornadas, y falsa en otro caso.
            model.Add(total_jornadas_trabajadas == 2).OnlyEnforceIf(doble_jornada)
            model.Add(total_jornadas_trabajadas != 2).OnlyEnforceIf(doble_jornada.Not())
            # Se guarda la variable en un diccionario previamente creado para su posterior acceso.
            dobles_por_trabajador[trabajador] = doble_jornada
            # Un trabajador que dobla jornadas solo puede hacerlo en las que está permitido (las de mañana y tarde)
            # Luego si doble_jornada es cierto, trabajará 0 turnos en las jornadas en las que no se puede doblar.
            model.Add(LinearExpr.Sum([
                asignacion.var
                for asignacion in asignaciones
                if asignacion.trabajador == trabajador
                and not asignacion.jornada.puede_doblar
            ]) == 0).OnlyEnforceIf(doble_jornada)

        else:
            # Si el trabajador no es voluntario para doble, solo podrá trabajar 1 jornada, sin más complicaciones.
            model.Add(total_jornadas_trabajadas <= 1)

    # Cada jornada debe cubrir su demanda.
    for puesto in puestos:
        for jornada in jornadas:
            model.Add(LinearExpr.Sum([
                asignacion.var
                for asignacion in asignaciones
                if asignacion.puesto == puesto
                and asignacion.jornada == jornada
            ]) == demanda.get((puesto, jornada), 0))

    # ******************************************************************************************************************
    # ************************************* FUNCIÓN OBJETIVO A MAXIMIZAR ***********************************************
    # ******************************************************************************************************************

    puntuacion_asignaciones: LinearExpr = LinearExpr.Sum([
        asignacion.puntuacion * asignacion.var
        for asignacion in asignaciones
    ])

    puntuacion_dobles: LinearExpr = LinearExpr.Sum([
        coeficientes_dobles[trabajador] * dobles_por_trabajador[trabajador]
        for trabajador in dobles_por_trabajador
        if coeficientes_dobles[trabajador] != 0
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

    resultado: set[tuple[Trabajador, PuestoTrabajo, Jornada]] = {
        (asignacion.trabajador, asignacion.puesto, asignacion.jornada)
        for asignacion in asignaciones
        if solver.Value(asignacion.var) == 1
    }

    trabajadores_asignados_dobles: set[Trabajador] = {
        trabajador
        for trabajador, _, _ in resultado
        if solver.Value(jornadas_trabajadas_por_trabajador[trabajador]) == 2
    }

    ultimo_codigo_asignado_por_especialidad: dict[PuestoTrabajo, int | None] = {
        puesto : max({
            trabajador.codigo
            for (trabajador, puesto, _) in resultado
            if trabajador in sets_especialidades.get(puesto, {})
        }, default=None)
        for puesto in puestos
    }

    ultimo_codigo_asignado_por_jornada: dict[TipoJornada, int | None] = {
        tipo_jornada : max({
            trabajador.codigo
            for (trabajador, _, jornada) in resultado
            if jornada in Jornada.jornadas_con_preferencia()
            and trabajador in set_preferencias_por_jornada[jornada.tipo_jornada]
        }, default=None)
        for tipo_jornada in TipoJornada
    }

    ultimo_codigo_voluntarios_doble: int | None = max({
        trabajador.codigo
        for trabajador in trabajadores_asignados_dobles
    }, default=None)


    if verbose.general:

        num_trabajadores_asignados: int = len({trabajador for trabajador, _, _ in resultado})
        num_trabajadores_disponibles: int = len({asignacion.trabajador for asignacion in asignaciones})

        num_preferencia_manana: int = len({
            asignacion.trabajador
            for asignacion in asignaciones
            if asignacion.trabajador in set_preferencias_por_jornada[TipoJornada.MANANA]
        })

        num_preferencia_tarde: int = len({
            asignacion.trabajador
            for asignacion in asignaciones
            if asignacion.trabajador in set_preferencias_por_jornada[TipoJornada.TARDE]
        })

        num_voluntarios_noche: int = len({
            asignacion.trabajador
            for asignacion in asignaciones
            if asignacion.trabajador in set_preferencias_por_jornada[TipoJornada.NOCHE]
        })

        num_voluntarios_dobles: int = len({
            asignacion.trabajador
            for asignacion in asignaciones
            if asignacion.trabajador in set_voluntarios_doble
        })

        puestos_demandados: int = 0
        puestos_demandados_por_jornada: dict[TipoJornada, int] = defaultdict(lambda: 0)
        for (_, jornada), valor in demanda.items():
            puestos_demandados += valor
            if jornada in Jornada.jornadas_con_preferencia():
                puestos_demandados_por_jornada[jornada.tipo_jornada] += valor

        puestos_demandados_manana: int = puestos_demandados_por_jornada[TipoJornada.MANANA]
        puestos_demandados_tarde: int = puestos_demandados_por_jornada[TipoJornada.TARDE]
        puestos_demandados_noche: int = puestos_demandados_por_jornada[TipoJornada.NOCHE]

        num_asignaciones_especialidad: int = solver.Value(total_asignaciones_especialidades)
        num_pref_manana_asignados_manana: int = solver.Value(total_asignaciones_respeta_jornada[TipoJornada.MANANA])
        num_pref_tarde_asignados_tarde: int = solver.Value(total_asignaciones_respeta_jornada[TipoJornada.TARDE])
        num_vol_noche_asignados_noche: int = solver.Value(total_asignaciones_respeta_jornada[TipoJornada.NOCHE])

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
        for trabajador, puesto, jornada in resultado:
            realiza_doble: str = 'DOBLE' if trabajador in trabajadores_asignados_dobles else ''
            polivalencia: str = trabajador.capacidades[puesto].nombre_es + (" ("+str(especialidades.get(puesto, []).index(trabajador))+")" if puesto in trabajador.especialidades and puesto in especialidades else "")
            preferencia: str = 'P.MAÑANA' + f' ({preferencias_jornada[TipoJornada.MANANA].index(trabajador)})' if trabajador in set_preferencias_por_jornada[TipoJornada.MANANA] else 'P.TARDE' + f' ({preferencias_jornada[TipoJornada.TARDE].index(trabajador)})' if trabajador in set_preferencias_por_jornada[TipoJornada.TARDE] else ''
            voluntario_noche: str = 'V.NOCHE' + f' ({preferencias_jornada[TipoJornada.NOCHE].index(trabajador)})' if trabajador in set_preferencias_por_jornada[TipoJornada.NOCHE] else ''
            voluntario_doble: str = 'V.DOBLE' + f' ({voluntarios_doble.index(trabajador)})' if trabajador in set_voluntarios_doble else ''
            puntuacion_por_asignacion: str = 'Punt: ' + str(coeficientes_asignaciones[trabajador, puesto, jornada] + solver.Value(dobles_por_trabajador.get(trabajador, 0)) * coeficientes_dobles.get(trabajador, 0))

            print(
                f"Trabajador {trabajador:<{3}} -> "
                f"{puesto.nombre_es:<{21}} | "
                f"{jornada.nombre_es:<{10}}"
                f"{realiza_doble:<{10}}{polivalencia:<{30}}"
                f"{preferencia:<{15}}"
                f"{voluntario_noche:<{15}}"
                f"{voluntario_doble:<{15}}"
                f"{puntuacion_por_asignacion:<{7}}"
            )


    if verbose.asignacion_puestos:
        if verbose.asignacion_trabajadores:
            print("\n\n")

        for puesto in puestos:
            print(f'{puesto.nombre_es}:')
            for jornada in jornadas:
                trabajadores_demandados: int = demanda.get((puesto, jornada), 0)
                trabajadores_asignados_a_puesto_y_jornada: int = len(
                    {trabajador for trabajador, p, j in resultado if p == puesto and j == jornada})
                if trabajadores_demandados != trabajadores_asignados_a_puesto_y_jornada:
                    print('**********************************************')
                    print(f'{puesto} en jornada {jornada}: MISMATCH!!!!')
                    print('**********************************************')
                if trabajadores_demandados != 0:
                    print('\t' f'{jornada.nombre_es} (demanda: {trabajadores_demandados}) asignado a {trabajadores_asignados_a_puesto_y_jornada} trabajadores:')

                    for trabajador in trabajadores:
                        if (trabajador, puesto, jornada) in resultado:
                            preferencia: str = 'P.MAÑANA' if trabajador in set_preferencias_por_jornada[TipoJornada.MANANA] else 'P.TARDE' if trabajador in set_preferencias_por_jornada[TipoJornada.TARDE] else ''
                            voluntario_noche: str = 'V.NOCHE' if trabajador in set_preferencias_por_jornada[TipoJornada.NOCHE] else ''
                            info_puesto: str = f'{puesto.nombre_es} | {trabajador.capacidades[puesto].nombre_es}' + (f' ({especialidades[puesto].index(trabajador)})' if trabajador in sets_especialidades[puesto] else '')
                            print(
                                f'\t\tTrabajador {trabajador:<{5}}'
                                f' -> '
                                f'{info_puesto:<{40}}'
                                f' {preferencia:<{15}}'
                                f'{voluntario_noche:<{15}}'
                            )

                            for p in trabajador.especialidades - {puesto}:
                                print("\t\t\t", end="")
                                puesto_info: str = f'{p.nombre_es}: {especialidades[p].index(trabajador)}'
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
    datos = data
    resultado = realizar_asignacion(
        *datos,
        verbose=Verbose(
            general=True,
            estadisticas_avanzadas=True,
            asignacion_puestos=False,
            asignacion_trabajadores=True
        )
    )