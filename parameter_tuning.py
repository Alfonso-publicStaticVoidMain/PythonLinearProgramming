from collections import defaultdict
from typing import Any

import optuna

from Clases import TipoJornada, Trabajador, PuestoTrabajo, Jornada
from asignacion import realizar_asignacion, Verbose, ParametrosPuntuacion
from parse import parse_all_data, ListasPreferencias


def compute_loss(
    asignacion: set[tuple[Trabajador, PuestoTrabajo, Jornada]],
    listas_preferencias: ListasPreferencias
) -> dict[str, int]:
    """
    Función que calcula una puntuación 'negativa' midiendo el número de veces que ocurrieron los siguientes sucesos:
        Se asignó un trabajador que no era voluntario a un turno de noche.
        No se respetó la preferencia de mañana o de tarde de un trabajador (se le asignó a otro tipo de jornada)
        No se asignó a un trabajador a su especialidad.
        No se asignó a un trabajador a su especialidad, pero sí que se asignó a otro trabajador con menor prioridad.
    """

    especialidades, voluntarios_noche, voluntarios_doble, preferencia_manana, preferencia_tarde = listas_preferencias

    no_voluntario_noche_asignados: int = 0
    preferencia_manana_tarde_no_respetadas: int = 0
    especialidad_no_asignada: int = 0

    for trabajador, puesto, jornada in asignacion:
        if puesto not in trabajador.especialidades:
            especialidad_no_asignada+=1
        if trabajador not in voluntarios_noche and jornada.tipo_jornada == TipoJornada.NOCHE:
            no_voluntario_noche_asignados+=1
        if trabajador in preferencia_manana and jornada.tipo_jornada != TipoJornada.MANANA:
            preferencia_manana_tarde_no_respetadas+=1
        if trabajador in preferencia_tarde and jornada.tipo_jornada != TipoJornada.TARDE:
            preferencia_manana_tarde_no_respetadas+=1

    return {
        "no_voluntario_noche_asignados": no_voluntario_noche_asignados,
        "preferencia_manana_tarde_no_respetadas": preferencia_manana_tarde_no_respetadas,
        "especialidad_no_asignada": especialidad_no_asignada,
    }


def objective(trial) -> int:
    coef_especialidad = trial.suggest_float("coef_especialidad", 0, 5)
    max_especialidad = trial.suggest_int("max_especialidad", -1000, 1000)
    decay_especialidad = trial.suggest_int("decay_especialidad", 0, 100)

    coef_capacidad = trial.suggest_float("coef_capacidad", 0, 5)
    max_capacidad = trial.suggest_int("max_capacidad", -1000, 1000)
    decay_capacidad = trial.suggest_int("decay_capacidad", 0, 100)

    coef_voluntarios_noche = trial.suggest_float("coef_voluntarios_noche", 0, 5)
    max_voluntarios_noche = trial.suggest_int("max_voluntarios_noche", -1000, 1000)
    decay_voluntarios_noche = trial.suggest_int("decay_voluntarios_noche", 0, 100)

    coef_voluntarios_doble = trial.suggest_float("coef_voluntarios_doble", 0, 5)
    max_voluntarios_doble = trial.suggest_int("max_voluntarios_doble", -1000, 1000)
    decay_voluntarios_doble = trial.suggest_int("decay_voluntarios_doble", 0, 100)

    coef_preferencia_jornada = trial.suggest_float("coef_preferencia_jornada", 0, 5)
    max_preferencia_jornada = trial.suggest_int("max_preferencia_jornada", -1000, 1000)
    decay_preferencia_jornada = trial.suggest_int("decay_preferencia_jornada", 0, 100)

    penalizacion_no_voluntario_noche = trial.suggest_int("penalizacion_no_voluntario_noche", 0, 1000)
    penalizacion_no_respeto_preferencia = trial.suggest_int("penalizacion_no_respeto_preferencia", 0, 1000)

    # Load your data once, outside or here if needed
    datos_trabajadores_puestos_jornadas, listas_preferencias, demanda, disponibilidad = parse_all_data()

    asignaciones = realizar_asignacion(
        datos_trabajadores_puestos_jornadas,
        listas_preferencias,
        demanda,
        disponibilidad,
        verbose=Verbose(False, False, False, False),
        parametros=ParametrosPuntuacion(
            coef_especialidad=coef_especialidad,
            max_especialidad=max_especialidad,
            decay_especialidad=decay_especialidad,

            coef_capacidad=coef_capacidad,
            max_capacidad=max_capacidad,
            decay_capacidad=decay_capacidad,

            coef_voluntarios_noche=coef_voluntarios_noche,
            max_voluntarios_noche=max_voluntarios_noche,
            decay_voluntarios_noche=decay_voluntarios_noche,

            coef_voluntarios_doble=coef_voluntarios_doble,
            max_voluntarios_doble=max_voluntarios_doble,
            decay_voluntarios_doble=decay_voluntarios_doble,

            coef_preferencia_jornada=coef_preferencia_jornada,
            max_preferencia_jornada=max_preferencia_jornada,
            decay_preferencia_jornada=decay_preferencia_jornada,

            penalizacion_no_voluntario_noche=penalizacion_no_voluntario_noche,
            penalizacion_no_respeto_preferencia=penalizacion_no_respeto_preferencia
        )
    )

    loss_dict: dict[str, int] = compute_loss(asignaciones, listas_preferencias)
    trial.set_user_attr("no_voluntario_noche_asignados", loss_dict["no_voluntario_noche_asignados"])
    trial.set_user_attr("preferencia_manana_tarde_no_respetadas", loss_dict["preferencia_manana_tarde_no_respetadas"])
    trial.set_user_attr("especialidad_no_asignada", loss_dict["especialidad_no_asignada"])
    #return 5 * loss_dict["no_voluntario_noche_asignados"] + 3 * loss_dict["preferencia_manana_tarde_no_respetadas"] + loss_dict["especialidad_no_asignada"]
    return sum(loss_dict.values())


def run_optimization(n_trials: int = 50) -> dict[str, Any]:
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    print("Mejores parámetros: ", study.best_params)
    print("Mejor pérdida: ", study.best_value)
    print("Desglose de pérdidas: ", study.best_trial.user_attrs)
    return study.best_params


run_optimization()