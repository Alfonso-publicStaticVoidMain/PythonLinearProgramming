from collections import defaultdict
from typing import Any

import optuna

from Clases import TipoJornada, Trabajador, PuestoTrabajo, Jornada
from asignacion import realizar_asignacion
from parse import parse_all_data


def compute_loss(
    asignacion: set[tuple[Trabajador, PuestoTrabajo, Jornada]],
    trabajadores: list[Trabajador],
    jornadas: list[Jornada],
    especialidades: dict[PuestoTrabajo, list[Trabajador]],
    voluntarios_noche: list[Trabajador],
    preferencia_manana: list[Trabajador],
    preferencia_tarde: list[Trabajador]
) -> dict[str, int]:
    """
    Función que calcula una puntuación 'negativa' midiendo el número de veces que ocurrieron los siguientes sucesos:
        Se asignó un trabajador que no era voluntario a un turno de noche.
        No se respetó la preferencia de mañana o de tarde de un trabajador (se le asignó a otro tipo de jornada)
        No se asignó a un trabajador a su especialidad.
        No se asignó a un trabajador a su especialidad, pero sí que se asignó a otro trabajador con menor prioridad.
    """
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
    # Suggest parameters (you can adjust ranges as needed)
    capacidad_base = trial.suggest_int("capacidad_base", 50, 150)
    capacidad_decaimiento = trial.suggest_int("capacidad_decaimiento", 5, 20)
    maximo_bonus_especialidad = trial.suggest_int("maximo_bonus_especialidad", 20, 100)
    bonus_maximo_jornada = trial.suggest_int("bonus_maximo_jornada", 5, 50)
    penalizacion_no_voluntario_noche = trial.suggest_int("penalizacion_no_voluntario_noche", 10, 60)

    # Load your data once, outside or here if needed
    trabajadores, puestos, jornadas, demanda, especialidades, voluntarios_noche, disponibilidad, voluntarios_doble, preferencia_manana, preferencia_tarde = parse_all_data()

    asignaciones = realizar_asignacion(
        trabajadores,
        puestos,
        jornadas,
        demanda,
        especialidades,
        voluntarios_noche,
        disponibilidad,
        voluntarios_doble,
        preferencia_manana,
        preferencia_tarde,
        verbose_estadisticas_avanzadas=False,
        verbose_asignacion_trabajadores=False,
        verbose_asignacion_puestos=False,
        capacidad_base=capacidad_base,
        capacidad_decaimiento=capacidad_decaimiento,
        maximo_bonus_especialidad=maximo_bonus_especialidad,
        bonus_maximo_jornada=bonus_maximo_jornada,
        penalizacion_no_voluntario_noche=penalizacion_no_voluntario_noche,
    )

    loss_dict: dict[str, int] = compute_loss(asignaciones, trabajadores, jornadas, especialidades, voluntarios_noche, preferencia_manana, preferencia_tarde)
    trial.set_user_attr("no_voluntario_noche_asignados", loss_dict["no_voluntario_noche_asignados"])
    trial.set_user_attr("preferencia_manana_tarde_no_respetadas", loss_dict["preferencia_manana_tarde_no_respetadas"])
    trial.set_user_attr("especialidad_no_asignada", loss_dict["especialidad_no_asignada"])
    return sum(loss_dict.values())


def run_optimization(n_trials: int = 50) -> dict[str, Any]:
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    print("Mejores parámetros: ", study.best_params)
    print("Mejor pérdida: ", study.best_value)
    print("Desglose de pérdidas: ", study.best_trial.user_attrs)
    return study.best_params


run_optimization(15)

# Primer test:
# Mejores parámetros: {'capacidad_base': 115, 'capacidad_decaimiento': 13, 'maximo_bonus_especialidad': 50, 'bonus_maximo_jornada': 5, 'penalizacion_no_voluntario_noche': 10}
# Mejor pérdida: 144.0
# Desglose de pérdidas: {'no_voluntario_noche_asignados': 32, 'preferencia_manana_tarde_no_respetadas': 60, 'especialidad_no_asignada': 52}

