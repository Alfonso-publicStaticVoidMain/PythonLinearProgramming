from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import NamedTuple

from frozendict import frozendict
from ortools.sat.cp_model_pb2 import CpSolverStatus
from ortools.sat.python import cp_model
from ortools.sat.python.cp_model import CpModel, IntVar, LinearExpr, CpSolver

from Clases import Trabajador, PuestoTrabajo, Jornada, TipoJornada
from parse import data, DatosTrabajadoresPuestosJornadas, ListasPreferencias
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


class Verbose(NamedTuple):
    estadisticas_avanzadas: bool
    general: bool
    asignacion_trabajadores: bool
    asignacion_puestos: bool


@dataclass(frozen=True, slots=True)
class ParametrosPuntuacion:
    """
    Clase para encapsular los parámetros que se van a utilizar en el cálculo de la función objetivo a maximizar por
    el modelo. Sus prefijos denotan el comportamiento que van a desempeñar en ese cálculo:
    <ul>
        <li>max: El máximo de puntuación que se puede asignar a esa categoría, previa multiplicación del coeficiente.</li>
        <li>decay: Decaimiento por cada posición que se aleja de la posición de máxima prioridad de la lista apropiada.</li>
    </ul>
    """

    max_especialidad: int = 300
    decay_especialidad: int = 1

    max_capacidad: int = 50
    decay_capacidad: int = 10

    max_voluntarios_doble: int = -700 # Este parámetro tiene un valor negativo para desincentivar que se asignen dobles
    decay_voluntarios_doble: int = 1

    max_preferencia_por_jornada: frozendict[TipoJornada, int] = frozendict({
        TipoJornada.MANANA : 300,
        TipoJornada.TARDE : 400,
        TipoJornada.NOCHE : 500,
    })

    decay_preferencia_por_jornada: frozendict[Jornada, int] = frozendict({
        TipoJornada.MANANA: 1,
        TipoJornada.TARDE: 1,
        TipoJornada.NOCHE: 1,
    })

    penalizacion_por_jornada: frozendict[Jornada, int] = frozendict({
        TipoJornada.MANANA: 0,
        TipoJornada.TARDE: 50,
        TipoJornada.NOCHE: 500,
    })

    def unpack(self: ParametrosPuntuacion) -> tuple:
        return (
            self.max_especialidad, self.decay_especialidad,
            self.max_capacidad, self.decay_capacidad,
            self.max_voluntarios_doble, self.decay_voluntarios_doble,
            self.max_preferencia_por_jornada, self.decay_preferencia_por_jornada, self.penalizacion_por_jornada
        )


def calcular_puntuacion(
    listas_preferencias: ListasPreferencias,
    vars: dict[tuple[Trabajador, PuestoTrabajo, Jornada], IntVar],
    parametros: ParametrosPuntuacion
) -> tuple[dict[tuple[Trabajador, PuestoTrabajo, Jornada], int], dict[Trabajador, float | int]]:
    puntuaciones: dict[tuple[Trabajador, PuestoTrabajo, Jornada], float | int] = {}
    dobles: dict[Trabajador, float | int] = {}

    especialidades, preferencia_por_jornada, voluntarios_doble = listas_preferencias

    (
        max_especialidad, decay_especialidad,
        max_capacidad, decay_capacidad,
        max_voluntarios_doble, decay_voluntarios_doble,
        max_preferencia_por_jornada, decay_preferencia_por_jornada, penalizacion_por_jornada
    ) = parametros.unpack()

    for trabajador in voluntarios_doble:
        dobles[trabajador] = max_voluntarios_doble - decay_voluntarios_doble * voluntarios_doble.index(trabajador)

    for trabajador, puesto, jornada in vars:

        # Puntuación por capacidad.
        puntuacion_capacidad: int = max(0, max_capacidad - decay_capacidad * (trabajador.capacidades[puesto].id - 1))

        # Puntuación por estar más alto en las listas de especialidades.
        puntuacion_especialidad: int = 0
        especialistas_puesto: list[Trabajador] = especialidades.get(puesto, [])
        if trabajador in especialistas_puesto:
            puntuacion_especialidad = max(0, max_especialidad - decay_especialidad * especialistas_puesto.index(trabajador))

        # Puntuación por preferencia de jornada o penalización por no ser voluntario para noche
        puntuacion_jornada: int = 0
        if jornada in [Jornada.MANANA, Jornada.TARDE, Jornada.NOCHE1, Jornada.NOCHE2]:
            tipo_jornada = jornada.tipo_jornada
            if trabajador in preferencia_por_jornada[tipo_jornada]:
                puntuacion_jornada += max_preferencia_por_jornada[tipo_jornada] - decay_preferencia_por_jornada[tipo_jornada] * preferencia_por_jornada[tipo_jornada].index(trabajador)
            else:
                puntuacion_jornada -= penalizacion_por_jornada[tipo_jornada]

        puntuaciones[trabajador, puesto, jornada] = puntuacion_capacidad + puntuacion_especialidad + puntuacion_jornada

    return puntuaciones, dobles


def realizar_asignacion(
    datos: DatosTrabajadoresPuestosJornadas,
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

    model: CpModel = CpModel()

    trabajadores, puestos, jornadas = datos
    especialidades, preferencia_por_jornada, voluntarios_doble = listas_preferencias

    # Se precomputan varios sets para comprobaciones de membresía más rápidas.
    # No se pueden recibir los datos como sets directamente porque el orden es importante para otras cosas.
    set_voluntarios_doble: set[Trabajador] = set(voluntarios_doble)
    set_preferencias_por_jornada: dict[TipoJornada, set[Trabajador]] = {
        tipo_jornada : set(lista_trabajadores)
        for tipo_jornada, lista_trabajadores in preferencia_por_jornada.items()
    }
    sets_especialidades: dict[PuestoTrabajo, set[Trabajador]] = {
        puesto : set(lista_trabajadores)
        for puesto, lista_trabajadores in especialidades.items()
    }

    # Se guardan las variables en un diccionario indexado por tuplas (trabajador, puesto, jornada)
    vars: dict[tuple[Trabajador, PuestoTrabajo, Jornada], IntVar] = {}

    # Se crean listas de variables que representan ciertos tipos de asignaciones con las que luego nos interesará
    # realizar ciertos cálculos: las asignaciones de un trabajador a una de sus especialidades, de voluntarios de
    # noche a turnos de noche y de preferencia de mañana o tarde a turnos de mañana o tarde respectivamente.
    asignaciones_especialidades: list[IntVar] = []
    asignaciones_respeta_jornada: dict[TipoJornada, list[IntVar]] = defaultdict(list)

    for trabajador in trabajadores:
        for puesto in puestos:
            for jornada in jornadas:
                # Solo se crean variables para las asignaciones permitidas, es decir, en las que el trabajador es
                # capaz de realizar el puesto y está disponible para esa jornada. Por lo tanto, se debe usar
                # vars.get((trabajador, puesto, jornada), 0) en vez de vars[trabajador, puesto, jornada].
                if puesto in trabajador.capacidades and (trabajador, jornada) in disponibilidad:
                    var = model.NewBoolVar(f'x_{trabajador}_{puesto}_{jornada}')
                    vars[trabajador, puesto, jornada] = var
                    if puesto in trabajador.especialidades:
                        asignaciones_especialidades.append(var)
                    if jornada in {Jornada.MANANA, Jornada.TARDE, Jornada.NOCHE1, Jornada.NOCHE2} and trabajador in set_preferencias_por_jornada[jornada.tipo_jornada]:
                        asignaciones_respeta_jornada[jornada.tipo_jornada].append(var)

    set_trabajadores_disponibles = {trabajador for (trabajador, _, _) in vars}
    num_trabajadores_disponibles: int = len(set_trabajadores_disponibles)

    # Se guardan expresiones lineales obtenidas al sumar las variables que previamente guardamos en ciertas listas.
    total_asignaciones_especialidades: LinearExpr = LinearExpr.Sum(asignaciones_especialidades)
    total_asignaciones_respeta_jornada: dict[TipoJornada, LinearExpr] = {
        tipo_jornada : LinearExpr.Sum(asignaciones)
        for tipo_jornada, asignaciones in asignaciones_respeta_jornada.items()
    }

    jornadas_trabajadas_por_trabajador: dict[Trabajador, LinearExpr] = {}
    dobles_por_trabajador: dict[Trabajador, IntVar] = {}

    for trabajador in trabajadores:
        # Se crea una expresión lineal que representa el total de jornadas trabajadas por el trabajador
        trabajador_capacidades = trabajador.capacidades
        total_jornadas_trabajadas: LinearExpr = LinearExpr.Sum([
            vars.get((trabajador, puesto, jornada), 0)
            for puesto in trabajador_capacidades
            for jornada in jornadas
        ])
        jornadas_trabajadas_por_trabajador[trabajador] = total_jornadas_trabajadas

        if trabajador in set_voluntarios_doble:
            # Cada trabajador voluntario para dobles puede trabajar a lo sumo 2 jornadas.
            model.Add(total_jornadas_trabajadas <= 2)

            for jornada in jornadas:
                # Cada trabajador solo puede desempeñar un puesto en cada jornada, no se puede dividir en dos.
                model.Add(LinearExpr.Sum([
                    vars.get((trabajador, puesto, jornada), 0)
                    for puesto in trabajador_capacidades
                ]) <= 1)

            # Se crea una variable nueva para representar si el trabajador en el que estamos actualmente iterando
            # realiza o no una jornada doble. Nótese que solo hacemos esto para los que son voluntarios para dobles.
            doble_jornada: IntVar = model.NewBoolVar(f'doble_jornada_{trabajador}')
            # Forzamos a que la variable doble_jornada sea verdadera cuando se trabajan 2 jornadas, y falsa en otro caso.
            model.Add(total_jornadas_trabajadas == 2).OnlyEnforceIf(doble_jornada)
            model.Add(total_jornadas_trabajadas != 2).OnlyEnforceIf(doble_jornada.Not())
            dobles_por_trabajador[trabajador] = doble_jornada
            # Un trabajador que dobla jornadas solo puede hacerlo en las que está permitido (las de mañana y tarde)
            # Luego si doble_jornada es cierto, trabajará 0 turnos en las jornadas en las que no se puede doblar.
            model.Add(LinearExpr.Sum([
                vars.get((trabajador, puesto, jornada), 0)
                for puesto in trabajador_capacidades
                for jornada in jornadas
                if not jornada.puede_doblar
            ]) == 0).OnlyEnforceIf(doble_jornada)

        else:
            # Si el trabajador no es voluntario para doble, solo podrá trabajar 1 jornada, sin más complicaciones.
            model.Add(total_jornadas_trabajadas <= 1)

    # Cada jornada debe cubrir su demanda.
    for puesto in puestos:
        for jornada in jornadas:
            model.Add(LinearExpr.Sum([
                vars.get((trabajador, puesto, jornada), 0)
                for trabajador in trabajadores
            ]) == demanda.get((puesto, jornada), 0))

    # Obtenemos de un método auxiliar los coeficientes a aplicar para calcular la puntuación de una asignación concreta
    puntuaciones: dict[tuple[Trabajador, PuestoTrabajo, Jornada], int]
    dobles: dict[Trabajador, int]
    puntuaciones, dobles = calcular_puntuacion(listas_preferencias, vars, parametros)
    model.Maximize(LinearExpr.Sum(
        LinearExpr.Sum([
            puntuaciones[trabajador, puesto, jornada] * vars[trabajador, puesto, jornada]
            for trabajador, puesto, jornada in vars
        ]),
        LinearExpr.Sum([
            dobles_por_trabajador[trabajador] * dobles[trabajador]
            for trabajador in dobles_por_trabajador
        ])
    ))

    solver: CpSolver = CpSolver()
    solver.parameters.num_search_workers = 8
    #solver.parameters.search_branching = cp_model.FIXED_SEARCH
    solver.parameters.max_time_in_seconds = 300.0
    solver.parameters.random_seed = 108
    solver.parameters.use_lns = True

    status: CpSolverStatus = solver.Solve(model)

    if verbose.general:
        if status == cp_model.OPTIMAL:
            print("Se encontró una solución óptima.")
        elif status == cp_model.FEASIBLE:
            print("Se encontró una solución factible, pero no necesariamente óptima.")
        else:
            print(f"No se encontraron soluciones.")

    if verbose.estadisticas_avanzadas:
        print_estadisticas_avanzadas(solver, "Estadísticas avanzadas")

    resultado = {
        (trabajador, puesto, jornada)
        for (trabajador, puesto, jornada) in vars
        if solver.Value(vars[trabajador, puesto, jornada]) == 1
    }

    trabajadores_asignados_dobles: set[Trabajador] = {
        trabajador
        for trabajador, _, _ in resultado
        if solver.Value(jornadas_trabajadas_por_trabajador[trabajador]) == 2
    }

    trabajadores_asignados: int = len({trabajador for trabajador, _, _ in resultado})

    puestos_demandados: int = 0
    puestos_demandados_por_jornada: dict[TipoJornada, int] = defaultdict(lambda:0)
    for (_, jornada), valor in demanda.items():  # type: tuple[PuestoTrabajo, Jornada], int
        puestos_demandados += valor
        if jornada in {Jornada.MANANA, Jornada.TARDE, Jornada.NOCHE1, Jornada.NOCHE2}:
            puestos_demandados_por_jornada[jornada.tipo_jornada] += valor

    ultimo_codigo_asignado_por_especialidad: dict[PuestoTrabajo, int | None] = {
        puesto: max({trabajador.codigo for (trabajador, puesto, _) in resultado if trabajador in sets_especialidades.get(puesto, {})}, default=None)
        for puesto in puestos
    }

    ultimo_codigo_voluntarios_doble: int | None = max({
        trabajador.codigo
        for trabajador in trabajadores_asignados_dobles
    }, default=None)


    if verbose.general:

        num_preferencia_manana: int = len({trabajador for (trabajador, _, _) in vars if trabajador in set_preferencias_por_jornada[TipoJornada.MANANA]})
        num_preferencia_tarde: int = len({trabajador for (trabajador, _, _) in vars if trabajador in set_preferencias_por_jornada[TipoJornada.TARDE]})
        num_voluntarios_noche: int = len({trabajador for (trabajador, _, _) in vars if trabajador in set_preferencias_por_jornada[TipoJornada.NOCHE]})

        num_asignaciones_especialidad: int = solver.Value(total_asignaciones_especialidades)
        num_pref_manana_asignados_manana: int = solver.Value(total_asignaciones_respeta_jornada[TipoJornada.MANANA])
        num_pref_tarde_asignados_tarde: int = solver.Value(total_asignaciones_respeta_jornada[TipoJornada.TARDE])
        num_vol_noche_asignados_noche: int = solver.Value(total_asignaciones_respeta_jornada[TipoJornada.NOCHE])

        print(f"Se demandaron {puestos_demandados_por_jornada[TipoJornada.MANANA]} puestos de mañana, {puestos_demandados_por_jornada[TipoJornada.TARDE]} puestos de tarde y {puestos_demandados_por_jornada[TipoJornada.NOCHE]} puestos nocturnos.")

        print(f"\nSe asignaron {num_vol_noche_asignados_noche} de {num_voluntarios_noche} ({100 * float(num_vol_noche_asignados_noche) / num_voluntarios_noche:.2f}%) voluntarios de noche a turnos de noche.")
        print(f"Se asignaron {num_pref_manana_asignados_manana} de {num_preferencia_manana} ({100 * float(num_pref_manana_asignados_manana / num_preferencia_manana):.2f}%) trabajadores con preferencia de mañana a turnos de mañana")
        print(f"Se asignaron {num_pref_tarde_asignados_tarde} de {num_preferencia_tarde} ({100 * float(num_pref_tarde_asignados_tarde / num_preferencia_tarde):.2f}%) trabajadores con preferencia de tarde a turnos de tarde")

        print(f'Número de asignaciones de especialidad alcanzado: {num_asignaciones_especialidad} de {puestos_demandados} ({100 * num_asignaciones_especialidad / puestos_demandados:.2f}%)')
        print(f'Número de trabajadores asignados: {trabajadores_asignados} de {num_trabajadores_disponibles} ({100 * float(trabajadores_asignados) / num_trabajadores_disponibles:.2f}%)')
        print(f'Puntuación alcanzada: {int(solver.ObjectiveValue())}\n')


    if verbose.asignacion_trabajadores:
        contador: int = 0
        for trabajador, puesto, jornada in resultado:
            contador += 1
            realiza_doble: str = 'DOBLE' if trabajador in trabajadores_asignados_dobles else ''
            polivalencia: str = trabajador.capacidades[puesto].nombre_es + (" ("+str(especialidades.get(puesto, []).index(trabajador))+")" if puesto in trabajador.especialidades and puesto in especialidades else "")
            preferencia: str = 'P.MAÑANA' + f' ({preferencia_por_jornada[TipoJornada.MANANA].index(trabajador)})' if trabajador in set_preferencias_por_jornada[TipoJornada.MANANA] else 'P.TARDE' + f' ({preferencia_por_jornada[TipoJornada.TARDE].index(trabajador)})' if trabajador in set_preferencias_por_jornada[TipoJornada.TARDE] else ''
            voluntario_noche: str = 'V.NOCHE' + f' ({preferencia_por_jornada[TipoJornada.NOCHE].index(trabajador)})' if trabajador in set_preferencias_por_jornada[TipoJornada.NOCHE] else ''
            voluntario_doble: str = 'V.DOBLE' + f' ({voluntarios_doble.index(trabajador)})' if trabajador in set_voluntarios_doble else ''
            puntuacion_por_asignacion: str = 'Punt: ' + str(puntuaciones[trabajador, puesto, jornada] + solver.Value(dobles_por_trabajador.get(trabajador, 0)) * dobles.get(trabajador, 0))

            print(
                f"{contador:<{3}} - "
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
                                f'\t\tTrabajador {trabajador:<5}'
                                f' -> '
                                f'{info_puesto:<40}'
                                f' {preferencia:<15}'
                                f'{voluntario_noche:<15}'
                            )

                            for p in trabajador.especialidades:
                                print("\t\t\t", end="")
                                puesto_info: str = f'{p.nombre_es}: {especialidades[p].index(trabajador)}'
                                print(f'{puesto_info:<15}')

    return resultado


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


if __name__ == "__main__":
    resultado1 = realizar_asignacion(
        *data,
        verbose=Verbose(
            general=True,
            estadisticas_avanzadas=True,
            asignacion_puestos=False,
            asignacion_trabajadores=True
        )
    )