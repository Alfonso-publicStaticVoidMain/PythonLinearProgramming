from __future__ import annotations

from collections import defaultdict
from dataclasses import Field
from dataclasses import dataclass, field, is_dataclass, fields
from datetime import date, datetime
from typing import ClassVar, Type, TypeVar, Any, Iterable, get_type_hints
from enum import Enum


T = TypeVar("T", bound="Identifiable")


def parse_data_into(cls: Type[T], data: dict[str, Any]) -> T:
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
    field_types: dict[str, Any] = {field.name: type_hints[field.name] for field in fields(cls)}
    filtered_data: dict[str, Any] = {}
    for k, v in data.items():
        if k in field_types:
            filtered_data[k] = v
        else:
            for field_name, field_type in field_types.items():
                if is_dataclass(field_type) and field_type.__name__ == k:
                    filtered_data[field_name] = parse_data_into(field_type, v)
                    break

    return cls(**filtered_data)


def get_by_id(collection: Iterable[T], cls: Type[T], id_: int | str | float, fallback: T | None = None) -> T | None:
    """
    Busca en un iterable de tipo T (que extienda Identificable) el único objeto con un cierto ID.
    :param collection: Iterable de tipo T, el cual debe tener extender de Identificable.
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
        for item in collection:
            if item.id == id_:
                cls.get_registro()[item.id] = item
                return item
        return fallback


def parse_bool(value: str | int) -> bool:
    return str(value).strip() in {'true', 'True', '1', 'yes', 'Yes', 'y', 'Y'}


@dataclass(eq=False, slots=True)
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
    def get_registro(cls: Type[T]) -> dict[int, T]:
        """
        Método de clase heredado por las clases que extiendan Identificable. Retorna el diccionario asignado a la propia
        clase en el diccionario _registro de Identificable. Si no estaba presente, primero lo inicializa como un
        diccionario vacío.
        """
        if cls not in cls._registros:
            cls._registros[cls] = {}
        return cls._registros[cls]

    def __post_init__(self: Identificable) -> None:
        """
        Añade automáticamente el objeto recién creado al registro si no estaba presente allí.
        Si ya estaba presente, comprueba si todos sus atributos son iguales. En tal caso, permite la creación del
        objeto, pero no lo intenta añadir al registro. Si algún atributo es distinto, lanza una excepción.
        """
        object.__setattr__(self, 'id', int(self.id))
        registro: dict[int, Identificable] = type(self).get_registro()
        existente: Identificable = registro.get(self.id, None)
        if existente is not None:
            for field_ in fields(self): # type: Field
                if field_.name.startswith('_'):
                    continue
                value_new: Any = getattr(self, field_.name)
                value_existing: Any = getattr(existente, field_.name)
                if value_new != value_existing:
                    raise ValueError(
                        f"Conflicto con ID duplicado en id={self.id} para la clase {type(self).__name__}: "
                        f"El campo '{field_.name}' es distinto (existente={value_existing!r}, nuevo={value_new!r})"
                    )
        else:
            registro[self.id] = self

    def __eq__(self: Identificable, other: object) -> bool:
        """
        Compara dos objetos, considerándolos iguales si el otro objeto es Identificable, es del mismo tipo que este, y
        tiene el mismo ID.
        """
        return isinstance(other, Identificable) and type(self) is type(other) and self.id == other.id

    def __hash__(self: Identificable) -> int:
        """
        Crea el hash del objeto teniendo en cuenta su tipo y su id.
        """
        return hash((type(self), self.id))

    def __format__(self: Trabajador, format_spec: str) -> str:
        return format(str(self), format_spec)

    @classmethod
    def from_id(cls: Type[T], id_: int | float | str) -> T | None:
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
            print(id_, "cannot be converted to an integer")
            print(e)

    @classmethod
    def get_or_create(cls: Type[T], data: dict[str, Any]) -> T:
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


@dataclass(eq=False, slots=True)
class Trabajador(Identificable):
    nombre: str
    apellidos: str
    codigo: int
    especialidades: set[PuestoTrabajo] = field(default_factory=set)
    capacidades: dict[PuestoTrabajo, NivelDesempeno] = field(default_factory=dict)

    def __post_init__(self: Trabajador) -> None:
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


@dataclass(eq=False, slots=True)
class PuestoTrabajo(Identificable):
    nombre_es: str

    def __post_init__(self: PuestoTrabajo) -> None:
        super(PuestoTrabajo, self).__post_init__()

    def __str__(self: PuestoTrabajo) -> str:
        return f'id={self.id} {self.nombre_es}'

    def __repr__(self: PuestoTrabajo) -> str:
        return f'PuestoTrabajo(id={self.id} {self.nombre_es})'


@dataclass(eq=False, slots=True)
class NivelDesempeno(Identificable):
    nombre_es: str

    def __post_init__(self: NivelDesempeno) -> None:
        super(NivelDesempeno, self).__post_init__()

    def __str__(self: NivelDesempeno) -> str:
        return f'{self.id}: {self.nombre_es}'

    def __repr__(self: NivelDesempeno) -> str:
        return f'NivelDesempeno({self.id}, {self.nombre_es})'


@dataclass(eq=False, slots=True)
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


@dataclass(eq=False, slots=True)
class Demanda(Identificable):
    fecha: date
    jornada_id: int
    jornada: Jornada
    escala_id: int
    muelle_id: int
    empresa_id: int
    operacion_id: int
    trafico_internacional: bool
    buque: str
    puestos_demandados: set[DemandasPuestosTrabajos] = field(default_factory=set)

    def __post_init__(self: Demanda) -> None:
        if isinstance(self.fecha, str):
            object.__setattr__(self, 'fecha', datetime.strptime(self.fecha, "%Y-%m-%d").date())
        object.__setattr__(self, 'jornada_id', int(self.jornada_id))
        object.__setattr__(self, 'jornada', Jornada.from_id(self.jornada_id))
        object.__setattr__(self, 'escala_id', int(self.escala_id))
        object.__setattr__(self, 'muelle_id', int(self.muelle_id))
        object.__setattr__(self, 'empresa_id', int(self.empresa_id))
        object.__setattr__(self, 'operacion_id', int(self.operacion_id))
        object.__setattr__(self, 'trafico_internacional', parse_bool(self.trafico_internacional))
        super(Demanda, self).__post_init__()


@dataclass(eq=False, slots=True)
class DemandasPuestosTrabajos(Identificable):
    num_trabajadores: int
    puesto_trabajo_id: int
    puesto_trabajo: PuestoTrabajo
    demanda_id: int
    demanda: Demanda
    fecha: date

    def __post_init__(self: DemandasPuestosTrabajos):
        object.__setattr__(self, 'puesto_trabajo_id', int(self.puesto_trabajo_id))
        object.__setattr__(self, 'puesto_trabajo', PuestoTrabajo.from_id(self.puesto_trabajo_id))
        object.__setattr__(self, 'demanda_id', int(self.demanda_id))
        object.__setattr__(self, 'demanda', Demanda.from_id(self.demanda_id))
        puestos_demandados: set[DemandasPuestosTrabajos] = self.demanda.puestos_demandados
        if self not in puestos_demandados: puestos_demandados.add(self)
        super(DemandasPuestosTrabajos, self).__post_init__()

class Contrato(Identificable):
    nombre_es: int

    def __post_init__(self: Contrato) -> None:
        super(Contrato, self).__post_init__()

class TrabajadorContrato(Identificable):
    trabajador_id: int
    trabajador: Trabajador
    contrato_id: int
    contrato: Contrato
    fecha_ini: date
    fecha_fin: date

    def __post_init__(self: TrabajadorContrato) -> None:
        object.__setattr__(self, 'trabajador_id', int(self.trabajador_id))
        object.__setattr__(self, 'contrato_id', int(self.contrato_id))
        object.__setattr__(self, 'trabajador', Trabajador.from_id(self.trabajador_id))
        if isinstance(self.fecha_ini, str):
            object.__setattr__(self, 'fecha_ini', datetime.strptime(self.fecha_ini, "%Y-%m-%d").date())
        if isinstance(self.fecha_fin, str):
            object.__setattr__(self, 'fecha_fin', datetime.strptime(self.fecha_fin, "%Y-%m-%d").date())
        super(TrabajadorContrato, self).__post_init__()

def test_trabajador_registry():
    print("Creating Trabajadores manually:")
    t1 = Trabajador(id=1, nombre="Alice", apellidos="Smith")
    t2 = Trabajador(id=2, nombre="Bob", apellidos="Jones")
    print(t1, t2)

    print("Registry contents:")
    for obj_id, obj in Trabajador.get_registro().items():
        print(f"ID {obj_id}: {obj}")

    print("Fetching from registry by id:")
    assert Trabajador.from_id(1) is t1
    assert Trabajador.from_id(2) is t2

    print("Using get_or_create_from with new data:")
    t3 = Trabajador.get_or_create({"id":3, "nombre":"Carol", "apellidos":"White"})
    print(t3)
    assert Trabajador.from_id(3) is t3

    print("Using get_or_create_from with existing id:")
    t1_again = Trabajador.get_or_create({"id":1, "nombre":"ShouldNotChange", "apellidos":"Name"})
    print(t1_again)
    assert t1_again is t1  # No overwrite

    print("Using get_or_create with new data:")
    t4 = Trabajador.get_or_create({"id":4, "nombre":"David", "apellidos":"Green"})
    print(t4)
    assert Trabajador.from_id(4) is t4

    print("Using get_or_create with existing id:")
    t2_again = Trabajador.get_or_create({"id":2, "nombre":"NewName", "apellidos":"LastName"})
    print(t2_again)
    assert t2_again is t2

    print("Registry contents:")
    for obj_id, obj in Trabajador.get_registro().items():
        print(f"ID {obj_id}: {obj}")

    print("All tests passed!")


if __name__ == "__main__":
    test_trabajador_registry()