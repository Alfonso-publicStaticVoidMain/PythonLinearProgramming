from collections import defaultdict

import optuna

from Clases import TipoJornada, Trabajador, PuestoTrabajo, Jornada
from asignacion import realizar_asignacion
from parse import parse_all_data


def compute_loss(
    assignments: dict[tuple[Trabajador, PuestoTrabajo, Jornada], bool],
    trabajadores: list[Trabajador],
    jornadas: list[Jornada],
    especialidades: dict[PuestoTrabajo, list[Trabajador]],
    voluntarios_noche: list[Trabajador],
    preferencia_manana: list[Trabajador],
    preferencia_tarde: list[Trabajador]
):
    no_voluntario_noche_asignados = 0
    preferencia_manana_tarde_no_respetadas = 0
    especialidad_no_asignada = 0
    especialista_sin_prioridad = 0

    jornadas_nocturnas: set[Jornada] = {j for j in jornadas if j.tipo_jornada == TipoJornada.NOCHE}
    jornadas_manana: set[Jornada] = {j for j in jornadas if j.tipo_jornada == TipoJornada.MANANA}
    jornadas_tarde: set[Jornada] = {j for j in jornadas if j.tipo_jornada == TipoJornada.TARDE}

    jornadas_asignadas: dict[Trabajador, list[Jornada]] = defaultdict(list)
    puestos_asignados: dict[Trabajador, list[PuestoTrabajo]] = defaultdict(list)

    for (trabajador, puesto, jornada), asignacion in assignments.items():
        if asignacion:
            jornadas_asignadas[trabajador].append(jornada)
            puestos_asignados[trabajador].append(puesto)

            if jornada in jornadas_nocturnas and trabajador not in voluntarios_noche:
                no_voluntario_noche_asignados += 1

            if jornada in jornadas_tarde and trabajador in preferencia_manana:
                preferencia_manana_tarde_no_respetadas += 1
            elif jornada in jornadas_manana and trabajador in preferencia_tarde:
                preferencia_manana_tarde_no_respetadas += 1

    for puesto, trabajadores_con_especialidad in especialidades.items():

        assigned_workers_to_puesto = set(t for t in trabajadores if any(
            assignments.get((t, puesto, j), False) for j in jornadas
        ))

        for t in trabajadores_con_especialidad:
            if t not in assigned_workers_to_puesto:
                especialidad_no_asignada += 1

        if especialidad_no_asignada > 0:
            for t in assigned_workers_to_puesto:
                if t not in trabajadores_con_especialidad:
                    especialista_sin_prioridad += 1

    # Weighted sum of loss
    loss = (10 * no_voluntario_noche_asignados +
            6 * preferencia_manana_tarde_no_respetadas +
            6 * especialidad_no_asignada +
            8 * especialista_sin_prioridad)

    return loss


def objective(trial):
    # Suggest parameters (you can adjust ranges as needed)
    capacidad_base = trial.suggest_int("capacidad_base", 50, 150)
    capacidad_decaimiento = trial.suggest_int("capacidad_decaimiento", 5, 20)
    maximo_bonus_especialidad = trial.suggest_int("maximo_bonus_especialidad", 20, 100)
    bonus_maximo_jornada = trial.suggest_int("bonus_maximo_jornada", 5, 50)
    penalizacion_no_voluntario_noche = trial.suggest_int("penalizacion_no_voluntario_noche", 10, 60)

    # Load your data once, outside or here if needed
    trabajadores, puestos, jornadas, demanda, capacidades_por_trabajador, especialidades, voluntarios_noche, disponibilidad, voluntarios_doble, preferencia_manana, preferencia_tarde = parse_all_data()

    assignments = realizar_asignacion(
        trabajadores, puestos, jornadas, demanda,
        capacidades_por_trabajador, especialidades,
        voluntarios_noche, disponibilidad, voluntarios_doble,
        preferencia_manana, preferencia_tarde,
        verbose_estadisticas_avanzadas=False,
        verbose_asignacion_trabajadores=False,
        verbose_asignacion_puestos=False,
        capacidad_base=capacidad_base,
        capacidad_decaimiento=capacidad_decaimiento,
        maximo_bonus_especialidad=maximo_bonus_especialidad,
        bonus_maximo_jornada=bonus_maximo_jornada,
        penalizacion_no_voluntario_noche=penalizacion_no_voluntario_noche,
    )

    loss = compute_loss(assignments, trabajadores, jornadas, especialidades, voluntarios_noche, preferencia_manana, preferencia_tarde)
    return loss  # minimize

def run_optimization(n_trials=50):
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    print("Best params:", study.best_params)
    print("Best loss:", study.best_value)
