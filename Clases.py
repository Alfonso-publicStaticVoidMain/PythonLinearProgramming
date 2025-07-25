from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field, is_dataclass, fields
from datetime import date, datetime
from enum import Enum
from typing import ClassVar, Type, TypeVar, Any, get_type_hints, Generic, Callable, cast, Mapping, ValuesView


@dataclass(eq=False, slots=True, frozen=True)
class Identificable:
    """
    Clase que actúa como superclase para todas las clases que se quiera que hereden un atributo id: int que las
    identifique de forma única. Posee también un atributo de clase _registros, un diccionario que mapea cada tipo de
    clase (que se espera que sea solamente de las que extiendan Identificable) a un diccionario que mapea cada posible
    id: int al objeto que tiene ese ID.
    """
    id: int
    _registros: ClassVar[dict[type, dict[int, Identificable]]] = defaultdict(dict)

    @classmethod
    def get_registro[T: Identificable](cls: type[T]) -> dict[int, T]:
        """
        Método de clase heredado por las clases que extiendan Identificable. Retorna el diccionario asignado a la propia
        clase en el diccionario _registro de Identificable.
        """
        return cls._registros[cls]

    def __new__(cls: type[Identificable], *args, **kwargs) -> Identificable:
        if 'id' in kwargs:
            id_ = int(kwargs['id'])
        elif args:
            id_ = int(args[0])
        else:
            raise ValueError(f"No se puede encontrar el argumento 'id' al crear un objeto de {cls.__name__} con parámetros {args} y {kwargs}")

        registro = cls._registros[cls]
        existente = registro.get(id_)

        if existente is not None:
            # Crea un objeto dummy para comparar atributos
            dummy = super(Identificable, cls).__new__(cls)
            object.__setattr__(dummy, 'id', id_)
            dummy.__init__(*args, **kwargs)

            for field_ in fields(cls):
                if field_.name.startswith('_'):
                    continue
                valor_existente = getattr(existente, field_.name)
                valor_nuevo = getattr(dummy, field_.name)
                if valor_existente != valor_nuevo:
                    raise ValueError(
                        f"Conflicto con ID duplicado en id={id_} para la clase {cls.__name__}: "
                        f"El campo '{field_.name}' es distinto "
                        f"(existente={valor_existente!r}, nuevo={valor_nuevo!r})"
                    )
            return existente
        else:
            obj = super(Identificable, cls).__new__(cls)
            registro[id_] = obj
            return obj

    def __post_init__(self: Identificable) -> None:
        """
        Fuerza que el ID del objeto Identificable sea un entero, previendo los casos en los que se le pudiera
        haber asignado un string.
        """
        object.__setattr__(self, 'id', int(self.id))

    def __format__(self: Trabajador, format_spec: str) -> str:
        return format(str(self), format_spec)

    @classmethod
    def from_id[T: Identificable](cls: type[T], id_: int | float | str) -> T | None:
        """
        Busca y retorna un objeto con el ID deseado, o None si no se encuentra.
        :param id_: ID del objeto a buscar.
        :return: El objeto guardado en el registro cuyo ID sea el mismo que el recibido como argumento, o None si
        no se encuentra tal objeto.
        """
        try:
            id_int = int(id_)
            return cls.get_registro().get(id_int)
        except (ValueError, TypeError) as e:
            print(id_, "no se puede convertir a un entero.")
            print(e)
            return None

    @classmethod
    def get_or_create[T: Identificable](cls: type[T], data: dict[str, Any]) -> T | None:
        """
        Crea un nuevo objeto obteniendo los datos de un diccionario recibido como argumento, o si un objeto con ese
        ID ya existía, retorna ese objeto.
        :param data: Diccionario del que obtener los datos.
        :return: Si ya existe en el registro un objeto con el ID asignado a la llave "id" del diccionario, retorna
        ese objeto. En otro caso, filtra los campos del diccionario quedándose solo con los que tengan nombres que
        coincidan con los nombres de los atributos de la clase y llama al constructor con esos valores, retornando
        el objeto construído de esta forma.
        """
        return cls.from_id(data["id"]) or parse_data_into(cls, data)

    def __int__(self: Identificable) -> int:
        return self.id

    def __eq__(self: Identificable, other: object) -> bool:
        """
        Compara dos objetos, considerándolos iguales si el otro objeto es Identificable, es del mismo tipo que este, y
        tiene el mismo ID.
        """
        return isinstance(other, Identificable) and type(self) is type(other) and self.id == other.id

    def __hash__(self: Identificable) -> int:
        """
        Crea el hash del objeto teniendo en cuenta su tipo y su ID.
        """
        return hash((type(self), self.id))


@dataclass(eq=False, slots=True, frozen=True)
class Trabajador(Identificable):
    nombre: str
    apellidos: str
    codigo: int
    especialidades: set[PuestoTrabajo] = field(default_factory=set)
    capacidades: dict[PuestoTrabajo, NivelDesempeno] = field(default_factory=dict)

    def __post_init__(self: Trabajador) -> None:
        object.__setattr__(self, 'codigo', int(self.codigo))
        super(Trabajador, self).__post_init__()

    def actualizar_capacidades(self: Trabajador, puesto: PuestoTrabajo, nivel: NivelDesempeno) -> None:
        """
        Actualiza el diccionario de capacidades del trabajador para reflejar el nivel de desempeño que tiene en el
        puesto, sustituyendo un valor previo si lo había. Si se añade un nivel de desempeño con ID 1, se añade el
        puesto al conjunto de especialidades, y si ya estaba en ese conjunto y se le sustituye su nivel de desempeño
        por uno que no sea el de especialidad, entonces se elimina del conjunto de especialidades.
        """
        self.capacidades[puesto] = nivel
        if nivel.id == 1 and puesto not in self.especialidades:
            self.especialidades.add(puesto)
        if nivel.id != 1 and puesto in self.especialidades:
            self.especialidades.remove(puesto)

    def __str__(self: Trabajador) -> str:
        return f'{self.codigo}'

    def __repr__(self: Trabajador) -> str:
        return f'Trabajador {self.id} - {self.nombre} {self.apellidos}'


@dataclass(eq=False, slots=True, frozen=True)
class PuestoTrabajo(Identificable):
    nombre_es: str

    def __post_init__(self: PuestoTrabajo) -> None:
        super(PuestoTrabajo, self).__post_init__()

    def __str__(self: PuestoTrabajo) -> str:
        return f'id={self.id} {self.nombre_es}'

    def __repr__(self: PuestoTrabajo) -> str:
        return f'PuestoTrabajo(id={self.id} {self.nombre_es})'


@dataclass(eq=False, slots=True, frozen=True)
class NivelDesempeno(Identificable):
    nombre_es: str

    def __post_init__(self: NivelDesempeno) -> None:
        super(NivelDesempeno, self).__post_init__()

    def __str__(self: NivelDesempeno) -> str:
        return f'{self.id}: {self.nombre_es}'

    def __repr__(self: NivelDesempeno) -> str:
        return f'NivelDesempeno({self.id}, {self.nombre_es})'


@dataclass(eq=False, slots=True, frozen=True)
class TrabajadorPuestoTrabajo(Identificable):
    trabajador: Trabajador
    puesto_trabajo: PuestoTrabajo
    nivel_desempeno: NivelDesempeno

    def __post_init__(self: TrabajadorPuestoTrabajo) -> None:
        self.trabajador.actualizar_capacidades(self.puesto_trabajo, self.nivel_desempeno)
        super(TrabajadorPuestoTrabajo, self).__post_init__()

    @property
    def get_trabajador_id(self: TrabajadorPuestoTrabajo) -> int:
        return self.trabajador.id

    @property
    def get_trabajador_nombre_y_apellidos(self: TrabajadorPuestoTrabajo) -> str:
        return self.trabajador.nombre + " " + self.trabajador.apellidos

    @property
    def get_puesto_trabajo_id(self: TrabajadorPuestoTrabajo) -> int:
        return self.puesto_trabajo.id

    @property
    def get_puesto_trabajo_nombre(self: TrabajadorPuestoTrabajo) -> str:
        return self.puesto_trabajo.nombre_es

    @property
    def nivel_desempeno_id(self: TrabajadorPuestoTrabajo) -> int:
        return self.nivel_desempeno.id

    @property
    def nivel_desempeno_nombre(self: TrabajadorPuestoTrabajo) -> str:
        return self.nivel_desempeno.nombre_es

"""
@dataclass(eq=False, slots=True, frozen=True)
class TipoJornada(Identificable):
    id: int
    nombre_es: str

    def __post_init__(self: TipoJornada) -> None:
        super(TipoJornada, self).__post_init__()

    def __str__(self: TipoJornada) -> str:
        return self.nombre_es


@dataclass(eq=False, slots=True, frozen=True)
class Jornada(Identificable):
    id: int
    nombre_es: str
    puede_doblar: bool
    tipo_jornada_id: int
    tipo_jornada: TipoJornada = field(default=None)

    def __post_init__(self: Jornada) -> None:
        object.__setattr__(self, 'puede_doblar', parse_bool(self.puede_doblar))
        object.__setattr__(self, 'tipo_jornada', TipoJornada.from_id(self.tipo_jornada_id))
        super(Jornada, self).__post_init__()

    def __str__(self: Jornada) -> str:
        return f"{self.nombre_es} ({self.tipo_jornada})"

    def __lt__(self: Jornada, other: Jornada) -> bool:
        if not isinstance(other, Jornada):
            return NotImplemented
        return self.id < other.id

    def __le__(self: Jornada, other: Jornada) -> bool:
        if not isinstance(other, Jornada):
            return NotImplemented
        return self.id <= other.id

    def __gt__(self: Jornada, other: Jornada) -> bool:
        if not isinstance(other, Jornada):
            return NotImplemented
        return self.id > other.id

    def __ge__(self: Jornada, other: Jornada) -> bool:
        if not isinstance(other, Jornada):
            return NotImplemented
        return self.id >= other.id

    @classmethod
    def jornadas_nocturnas(cls: Jornada) -> set[Jornada]:
        return {
            jornada
            for jornada in cls.get_registro().values()
            if jornada.tipo_jornada.nombre_es == "Noche"
        }

    @classmethod
    def jornadas_puede_doblar(cls: Jornada) -> set[Jornada]:
        return {
            jornada
            for jornada in Jornada.get_registro().values()
            if jornada.puede_doblar
        }

    @classmethod
    def jornadas_con_preferencia(cls: Jornada) -> set[Jornada]:
        return {
            jornada
            for jornada in Jornada.get_registro().values()
            if jornada.nombre_es != "Partida"
        }
"""

class TipoJornada(Enum):
    MANANA = (1, "Mañana")
    TARDE = (2, "Tarde")
    NOCHE = (3, "Noche")

    id: int
    nombre_es: str

    def __init__(self: TipoJornada, id: int, nombre_es: str) -> None:
        self.id = id
        self.nombre_es = nombre_es

    def __str__(self: TipoJornada) -> str:
        return self.nombre_es

    @classmethod
    def from_id(cls: Type[TipoJornada], id_: int | str) -> TipoJornada | None:
        try:
            id_ = int(id_)
            for tipo_jornada in TipoJornada:
                if tipo_jornada.id == id_:
                    return tipo_jornada
        except (ValueError, TypeError) as e:
            print(id_, "cannot be converted to an integer")
            print(e)
            return None


class Jornada(Enum):
    MANANA = (1, "Mañana", True, TipoJornada.MANANA)
    TARDE = (2, "Tarde", True, TipoJornada.TARDE)
    PARTIDA = (3, "Partida", False, TipoJornada.MANANA)
    NOCHE1 = (4, "Noche1", False, TipoJornada.NOCHE)
    NOCHE2 = (5, "Noche2", False, TipoJornada.NOCHE)

    def __init__(self: Jornada, id: int, nombre_es: str, puede_doblar: bool, tipo_jornada: TipoJornada) -> None:
        self.id = id
        self.nombre_es = nombre_es
        self.puede_doblar = puede_doblar
        self.tipo_jornada = tipo_jornada

    def __str__(self: Jornada) -> str:
        return f"{self.nombre_es} ({self.tipo_jornada})"

    def __lt__(self: Jornada, other: Jornada) -> bool:
        if not isinstance(other, Jornada):
            return NotImplemented
        return self.id < other.id

    def __le__(self: Jornada, other: Jornada) -> bool:
        if not isinstance(other, Jornada):
            return NotImplemented
        return self.id <= other.id

    def __gt__(self: Jornada, other: Jornada) -> bool:
        if not isinstance(other, Jornada):
            return NotImplemented
        return self.id > other.id

    def __ge__(self: Jornada, other: Jornada) -> bool:
        if not isinstance(other, Jornada):
            return NotImplemented
        return self.id >= other.id

    @classmethod
    def jornadas_nocturnas(cls: Jornada) -> set[Jornada]:
        return {
            jornada
            for jornada in Jornada
            if jornada.tipo_jornada == TipoJornada.NOCHE
        }

    @classmethod
    def jornadas_puede_doblar(cls: Jornada) -> set[Jornada]:
        return {jornada for jornada in Jornada if jornada.puede_doblar}

    @classmethod
    def jornadas_con_preferencia(cls: Jornada) -> set[Jornada]:
        return {Jornada.MANANA, Jornada.TARDE, Jornada.NOCHE1, Jornada.NOCHE2}

    @classmethod
    def from_id(cls: Type[Jornada], id_: int | str) -> Jornada | None:
        try:
            id_ = int(id_)
            for jornada in Jornada:
                if jornada.id == id_:
                    return jornada
        except (ValueError, TypeError) as e:
            print(id_, "cannot be converted to an integer")
            print(e)
            return None


def parse_data_into[T](cls: type[T], data: dict[str, Any]) -> T:
    """
    Parsea el contenido de un diccionario en una clase, asignándole a sus atributos los valores de las llaves del
    diccionario cuyo nombre coincida exactamente con el nombre del atributo.

    Si la clase posee atributos adicionales, estos quedarán sin asignar, y si los datos poseen llaves que
    no aparecen en los atributos, esos datos no se asignarán a nada.

    Las llaves cuyo nombre coincida con una anotación de tipo de los atributos de la clase serán recursivamente parseados
    en ese atributo.

    :param cls: Clase de la que crear el objeto.
    :param data: Diccionario conteniendo los valores de los atributos.
    :return: Un objeto de la clase cls cuyos atributos toman como valor los valores de las llaves del diccionario que
    coincidan con los nombres de esos atributos, o un parse recursivo de los valores de las llaves cuyo nombre
    coincida con las anotaciones de tipo de esos atributos.
    """

    if not is_dataclass(cls):
        raise ValueError(f"{cls.__name__} is not a dataclass")

    type_hints: dict[str, Any] = get_type_hints(cls)
    field_names_and_types: dict[str, Any] = {
        field.name : type_hints[field.name]
        for field in fields(cls)
    }
    filtered_data: dict[str, Any] = {}
    for k, v in data.items():
        if k in field_names_and_types:
            filtered_data[k] = v
        else:
            for field_name, field_type in field_names_and_types.items():
                if is_dataclass(field_type) and field_type.__name__ == k:
                    filtered_data[field_name] = parse_data_into(field_type, v)
                    break

    return cls(**filtered_data)


def get_by_id[T](
    collection: Iterable[T],
    cls: type[T],
    id_: int | str | float,
    fallback: T | None = None
) -> T | None:
    """
    Busca en un iterable de tipo T (que extienda Identificable) el único objeto con un cierto ID.
    :param collection: Objeto iterable que contiene objetos de tipo T, el cual debe extender de Identificable.
    :param cls: Clase T contenida en el iterable.
    :param id_: ID que buscar dentro del iterable.
    :param fallback: Valor por defecto que retornar si no se encuentra el ID. Por defecto es None.
    :return: El único objeto del iterable con el ID recibido como argumento, o fallback si no se encuentra.

    Para maximizar la eficiencia del algoritmo, primero busca con orden O(1) el objeto en el _registro global, y
    comprueba si el resultado obtenido está en la colección. Si es así, lo retorna. En otro caso, busca manualmente en la
    colección por un item con el ID recibido como argumento, y si lo encuentra, lo añade al registro y lo retorna.
    Si no lo encuentra, retorna fallback.
    """
    candidate: T = cls.from_id(id_)
    if candidate in collection:
        return candidate
    else:
        for item in collection: # type: T
            if item.id == id_:
                cls.get_registro()[item.id] = item
                return item
        return fallback


def parse_bool(value: str | int) -> bool:
    return str(value).strip() in {'true', 'True', '1', 'yes', 'Yes', 'y', 'Y'}


if __name__ == "__main__":
    t1: Trabajador = Trabajador(id=1, nombre="Alfonso", apellidos="Gallego", codigo=5000)
    t2: Trabajador = Trabajador(id=2, nombre="Pepito", apellidos="Pérez", codigo=5001)
    t3: Trabajador = Trabajador(id=3, nombre="María", apellidos="Gómez", codigo=5002)
    t4: Trabajador = Trabajador(id=4, nombre="José Manuel", apellidos="Fernández", codigo=5003)
    t5: Trabajador = Trabajador(id=5, nombre="Ana", apellidos="Díaz", codigo=5004)

    for id, t in Trabajador.get_registro().items():  # type: int, Trabajador
        print(t.nombre, t.apellidos, f"id={id}")
