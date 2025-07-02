from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

from frozendict import frozendict

from Clases import TipoJornada, Jornada, PuestoTrabajo, Trabajador


class Verbose(NamedTuple):
    estadisticas_avanzadas: bool
    general: bool
    asignacion_trabajadores: bool
    asignacion_puestos: bool


class DatosTrabajadoresPuestosJornadas(NamedTuple):
    trabajadores: list[Trabajador]
    puestos: list[PuestoTrabajo]
    jornadas: list[Jornada]


class ListasPreferencias(NamedTuple):
    especialidades: dict[PuestoTrabajo, list[Trabajador]]
    preferencias_jornada: dict[TipoJornada, list[Trabajador]]
    voluntarios_doble: list[Trabajador]


class IndicesPreferencias(NamedTuple):
    especialidades: dict[tuple[PuestoTrabajo, Trabajador], int]
    preferencias_jornada: dict[tuple[TipoJornada, Trabajador], int]
    voluntarios_doble: dict[Trabajador, int]

    @classmethod
    def from_listas(cls: IndicesPreferencias, listas: ListasPreferencias) -> IndicesPreferencias:
        especialidades: dict[tuple[PuestoTrabajo, Trabajador], int] = {
            (puesto, trabajador) : listas.especialidades[puesto].index(trabajador)
            for puesto in listas.especialidades
            for trabajador in listas.especialidades[puesto]
        }

        preferencias_jornada: dict[tuple[TipoJornada, Trabajador], int] = {
            (tipo_jornada, trabajador) : listas.preferencias_jornada[tipo_jornada].index(trabajador)
            for tipo_jornada in listas.preferencias_jornada
            for trabajador in listas.preferencias_jornada[tipo_jornada]
        }

        voluntarios_doble: dict[Trabajador, int] = {
            trabajador : listas.voluntarios_doble.index(trabajador)
            for trabajador in listas.voluntarios_doble
        }

        return IndicesPreferencias(especialidades, preferencias_jornada, voluntarios_doble)


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

    max_especialidad: int = 500
    decay_especialidad: int = 5

    max_capacidad: int = 50
    decay_capacidad: int = 10

    max_voluntarios_doble: int = -700 # Este parámetro tiene un valor negativo para desincentivar que se asignen dobles
    decay_voluntarios_doble: int = 1

    max_preferencia_por_jornada: frozendict[TipoJornada, int] = frozendict({
        TipoJornada.MANANA : 300,
        TipoJornada.TARDE : 400,
        TipoJornada.NOCHE : 500,
    })

    decay_preferencia_por_jornada: frozendict[TipoJornada, int] = frozendict({
        TipoJornada.MANANA: 1,
        TipoJornada.TARDE: 1,
        TipoJornada.NOCHE: 1,
    })

    penalizacion_por_jornada: frozendict[TipoJornada, int] = frozendict({
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

    def unpack_festivo(self):
        return (
            self.max_capacidad, self.decay_capacidad,
            self.max_voluntarios_doble, self.decay_voluntarios_doble
        )