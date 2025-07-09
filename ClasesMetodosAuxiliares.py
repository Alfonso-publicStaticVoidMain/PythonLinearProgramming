from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple
from frozendict import frozendict
from ortools.sat.python.cp_model import IntVar, CpSolver

from Clases import TipoJornada, Jornada, PuestoTrabajo, Trabajador


class IdsAsignacion(NamedTuple):
    trabajador_id: int
    puesto_id: int
    jornada_id: int
    var: IntVar
    puntuacion: int

    def get_trabajador(self: IdsAsignacion) -> Trabajador:
        return Trabajador.from_id(self.trabajador_id)

    def get_puesto(self: IdsAsignacion) -> PuestoTrabajo:
        return PuestoTrabajo.from_id(self.puesto_id)

    def get_jornada(self: IdsAsignacion) -> Jornada:
        return Jornada.from_id(self.jornada_id)


class Asignacion(NamedTuple):
    trabajador: Trabajador
    puesto: PuestoTrabajo
    jornada: Jornada
    var: IntVar
    puntuacion: int


class Verbose(NamedTuple):
    estadisticas_avanzadas: bool
    general: bool
    asignacion_trabajadores: bool
    asignacion_puestos: bool


class DatosTrabajadoresPuestosJornadas(NamedTuple):
    trabajadores: list[Trabajador]
    puestos: list[PuestoTrabajo]
    jornadas: list[Jornada]


class IdsTrabajadoresPuestosJornadas(NamedTuple):
    trabajadores: list[int]
    puestos: list[int]
    jornadas: list[int]


class ListasPreferencias(NamedTuple):
    especialidades: dict[PuestoTrabajo, list[Trabajador]]
    preferencias_jornada: dict[TipoJornada, list[Trabajador]]
    voluntarios_doble: list[Trabajador]


class IdsListasPreferencias(NamedTuple):
    especialidades: dict[int, list[int]]
    preferencias_jornada: dict[int, list[int]]
    voluntarios_doble: list[int]


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


'''
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

    max_voluntarios_doble: int = -1000 # Este parámetro tiene un valor negativo para desincentivar que se asignen dobles
    decay_voluntarios_doble: int = 1

    max_preferencia_por_jornada: frozendict[TipoJornada, int] = frozendict({
        TipoJornada.MANANA : 300,
        TipoJornada.TARDE : 500,
        TipoJornada.NOCHE : 700,
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

    def unpack_festivo(self: ParametrosPuntuacion) -> tuple:
        return (
            self.max_capacidad, self.decay_capacidad,
            self.max_voluntarios_doble, self.decay_voluntarios_doble
        )
'''


@dataclass(frozen=True, slots=True)
class IdsParametrosPuntuacion:
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

    max_voluntarios_doble: int = -1000 # Este parámetro tiene un valor negativo para desincentivar que se asignen dobles
    decay_voluntarios_doble: int = 1

    max_preferencia_por_jornada: frozendict[int, int] = frozendict({
        1 : 300,
        2 : 500,
        3 : 700,
    })

    decay_preferencia_por_jornada: frozendict[int, int] = frozendict({
        1 : 1,
        2 : 1,
        3 : 1,
    })

    penalizacion_por_jornada: frozendict[int, int] = frozendict({
        1 : 0,
        2 : 50,
        3 : 500,
    })

    def unpack(self: ParametrosPuntuacion) -> tuple:
        return (
            self.max_especialidad, self.decay_especialidad,
            self.max_capacidad, self.decay_capacidad,
            self.max_voluntarios_doble, self.decay_voluntarios_doble,
            self.max_preferencia_por_jornada, self.decay_preferencia_por_jornada, self.penalizacion_por_jornada
        )

    def unpack_festivo(self: ParametrosPuntuacion) -> tuple:
        return (
            self.max_capacidad, self.decay_capacidad,
            self.max_voluntarios_doble, self.decay_voluntarios_doble
        )



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

    max_voluntarios_doble: int = -1000 # Este parámetro tiene un valor negativo para desincentivar que se asignen dobles
    decay_voluntarios_doble: int = 1

    max_preferencia_por_jornada: frozendict[TipoJornada, int] = frozendict({
        TipoJornada.from_id(1) : 300,
        TipoJornada.from_id(2) : 500,
        TipoJornada.from_id(3) : 700,
    })

    decay_preferencia_por_jornada: frozendict[TipoJornada, int] = frozendict({
        TipoJornada.from_id(1) : 1,
        TipoJornada.from_id(2) : 1,
        TipoJornada.from_id(3) : 1,
    })

    penalizacion_por_jornada: frozendict[TipoJornada, int] = frozendict({
        TipoJornada.from_id(1) : 0,
        TipoJornada.from_id(2) : 50,
        TipoJornada.from_id(3) : 500,
    })

    def unpack(self: ParametrosPuntuacion) -> tuple:
        return (
            self.max_especialidad, self.decay_especialidad,
            self.max_capacidad, self.decay_capacidad,
            self.max_voluntarios_doble, self.decay_voluntarios_doble,
            self.max_preferencia_por_jornada, self.decay_preferencia_por_jornada, self.penalizacion_por_jornada
        )

    def unpack_festivo(self: ParametrosPuntuacion) -> tuple:
        return (
            self.max_capacidad, self.decay_capacidad,
            self.max_voluntarios_doble, self.decay_voluntarios_doble
        )

def print_estadisticas_avanzadas(solver: CpSolver, mensaje: str = ""):
    print(mensaje)
    print(f"Problema resuelto en: {formatear_tiempo(solver.wall_time)}")
    print(f"Conflictos: {solver.NumConflicts()}")
    print(f"Ramas: {solver.NumBranches()}\n")


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


def formatear_tiempo(segundos: float | int) -> str:
    if segundos < 10:
        return f"{segundos:.4f} segundos"
    segundos_int = int(segundos)
    if segundos_int < 60:
        return f"{segundos_int} segundos"
    elif segundos_int < 3600:
        minutos = segundos_int // 60
        segundos_restantes = segundos_int % 60
        return f"{minutos} minutos y {segundos_restantes} segundos"
    else:
        horas = segundos_int // 3600
        segundos_restantes = segundos_int % 3600
        minutos = segundos_restantes // 60
        segundos_restantes = segundos_restantes % 60
        return f"{horas} horas, {minutos} minutos y {segundos_restantes} segundos"
