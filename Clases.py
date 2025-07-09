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
    def get_registro(cls: type[T]) -> dict[int, T]:
        """
        Método de clase heredado por las clases que extiendan Identificable. Retorna el diccionario asignado a la propia
        clase en el diccionario _registro de Identificable.
        """
        return cls._registros[cls]

    def __new__(cls: type[T], *args, **kwargs) -> T:
        if 'id' in kwargs:
            id_ = int(kwargs['id'])
        elif args:
            id_ = int(args[0])
        else:
            raise ValueError(f"No se puede encontrar el argumento 'id' al crear un objeto de {cls.name} con parámetros {args} y {kwargs}")

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
    def from_id(cls: type[T], id_: int | float | str) -> T | None:
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
    def get_or_create(cls: type[T], data: dict[str, Any]) -> T | None:
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
    especialidades_ids: set[int] = field(default_factory=set)
    capacidades: dict[PuestoTrabajo, NivelDesempeno] = field(default_factory=dict)
    capacidades_ids: dict[int, int] = field(default_factory=dict)

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
        self.capacidades_ids[puesto.id] = nivel.id
        if nivel.id == 1 and puesto not in self.especialidades:
            self.especialidades.add(puesto)
            self.especialidades_ids.add(puesto.id)
        if nivel.id != 1 and puesto in self.especialidades:
            self.especialidades.remove(puesto)
            self.especialidades_ids.remove(puesto.id)

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


@dataclass(eq=False, slots=True, frozen=True)
class TipoJornada(Identificable):
    id: int
    nombre_es: str

    manana_Bilbao_id: ClassVar[int] = 1
    tarde_Bilbao_id: ClassVar[int] = 2
    noche_Bilbao_id: ClassVar[int] = 3

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

    manana_Bilbao_id: ClassVar[int] = 1
    tarde_Bilbao_id: ClassVar[int] = 2
    partida_Bilbao_id: ClassVar[int] = 3
    noche1_Bilbao_id: ClassVar[int] = 4
    noche2_Bilbao_id: ClassVar[int] = 5

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
    def jornadas_nocturnas(cls: Jornada) -> set[int]:
        return {
            jornada_id
            for jornada_id in Jornada.get_registro().keys()
            if Jornada.from_id(jornada_id).tipo_jornada.nombre_es == "Noche"
        }

    @classmethod
    def jornadas_puede_doblar(cls: Jornada) -> set[int]:
        return {
            jornada_id
            for jornada_id in Jornada.get_registro().keys()
            if Jornada.from_id(jornada_id).puede_doblar
        }

    @classmethod
    def jornadas_con_preferencia(cls: Jornada) -> set[int]:
        return {
            jornada_id
            for jornada_id in Jornada.get_registro().keys()
            if Jornada.from_id(jornada_id).nombre_es != "Partida"
        }

    @classmethod
    def mañana_Bilbao(cls: Jornada) -> Jornada:
        return Jornada(1, "Mañana", True, 1)

    @classmethod
    def tarde_Bilbao(cls: Jornada) -> Jornada:
        return Jornada(2, "Tarde", True, 2)

    @classmethod
    def partida_Bilbao(cls: Jornada) -> Jornada:
        return Jornada(3, "Partida", False, 1)

    @classmethod
    def noche1_Bilbao(cls: Jornada) -> Jornada:
        return Jornada(4, "Noche1", False, 3)

    @classmethod
    def noche2_Bilbao(cls: Jornada) -> Jornada:
        return Jornada(5, "Noche2", False, 3)

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

"""

T = TypeVar("T", Identificable, Jornada, TipoJornada)


def parse_data_into(cls: type[T], data: dict[str, Any]) -> T:
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


def get_by_id(
    collection: Iterable[T],
    cls: Type[T],
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


_V = TypeVar("V")
K = TypeVar("K", bound=Identificable)
V = TypeVar("V", bound=Identificable)


class IdDict(dict[int, _V], Generic[T, _V]):

    cls: type[T] | None

    def __init__(
        self: IdDict[T, _V],
        items: Mapping[int | str | T, _V] | Iterable[tuple[int | str | T, _V]] = (),
        cls: type[T] | None = None
    ) -> None:
        if cls is not None and not issubclass(cls, Identificable):
            raise ValueError(f"El tipo {cls} no extiende de Identificable.")

        self.cls = cls

        if isinstance(items, Mapping):
            items = items.items()
        elif not isinstance(items, Iterable):
            raise TypeError("items debe ser un mapeo o un iterable de pares (key, value)")

        result: dict[int, _V] = {}
        for key, value in items:
            if isinstance(key, Identificable):
                if self.cls is None:
                    self.cls = type(key)
                elif not isinstance(key, self.cls):
                    raise ValueError(f"Todos los objetos deben ser del mismo tipo Identificable ({self.cls}).")
                key_id = int(key)
            elif isinstance(key, int):
                key_id = key
            elif isinstance(key, str):
                try:
                    key_id = int(key)
                    if key_id < 0:
                        raise ValueError()
                except ValueError:
                    raise ValueError(f"Clave inválida: {key}")
            else:
                raise TypeError(f"Clave inválida: {key} (tipo {type(key)})")

            result[key_id] = value

        if self.cls is None:
            raise ValueError("No se pudo inferir el tipo cls a partir de las claves proporcionadas.")

        registro = self.cls.get_registro()
        for key_id in result:
            if key_id not in registro:
                raise ValueError(f"{key_id} no está presente en el registro de la clase {self.cls}")

        super().__init__(result)

    def __setitem__(self, key: int | str | T, value: _V) -> None:
        key_id = validate_id(key, self.cls)
        super().__setitem__(key_id, value)

    def keys_ids(self) -> set[int]:
        return set(super().keys())

    def keys(self) -> IdList[T]:
        return IdList(self.keys_ids(), cls=self.cls)

    def keys_objects(self) -> set[T]:
        return {self.cls.from_id(id_) for id_ in self.keys_ids()}

    def items(self) -> set[tuple[int, _V]]:
        return set(super().items())

    def values(self) -> ValuesView[_V]:
        return super().values()


class IdDoubleDict(dict[int, int], Generic[K, V]):

    key_cls: type[K]
    value_cls: type[V]

    def __init__(
        self: IdDoubleDict[K, V],
        items: Mapping[int | str | K, int | str | V] | Iterable[tuple[int | str | K, int | str | V]] = (),
        key_cls: type[K] | None = None,
        value_cls: type[V] | None = None
    ) -> None:
        if key_cls is not None and not issubclass(key_cls, Identificable):
            raise ValueError(f"El tipo {key_cls} no extiende de Identificable.")
        if value_cls is not None and not issubclass(value_cls, Identificable):
            raise ValueError(f"El tipo {value_cls} no extiende de Identificable.")

        if isinstance(items, Mapping):
            items = items.items()
        elif not isinstance(items, Iterable):
            raise TypeError("items debe ser un mapeo o un iterable de pares (key, value)")

        result: dict[int, int] = {}

        for key, value in items:
            # Procesado de las llaves (keys)
            if isinstance(key, Identificable):
                # Si el atributo key_cls era nulo, se infiere de la clase de la llave en la que se está itereando
                if key_cls is None:
                    key_cls = type(key)
                # Si key_cls no era nulo, se comprueba que la llave tiene ese tipo.
                elif not isinstance(key, key_cls):
                    raise ValueError(f"Todos los objetos clave deben ser del mismo tipo Identificable ({key_cls}).")
                key_id = int(key)
            elif isinstance(key, str):
                try:
                    key_id = int(key)
                    if key_id < 0:
                        raise ValueError()
                except ValueError:
                    raise ValueError(f"Clave inválida: {key}")
            elif isinstance(key, int):
                key_id = key
            else:
                raise TypeError(f"Clave inválida: {key} (tipo {type(key)})")

            # Procesado de los valores (values)
            if isinstance(value, Identificable):
                if value_cls is None:
                    value_cls = type(value)
                elif not isinstance(value, value_cls):
                    raise ValueError(f"Todos los valores deben ser del mismo tipo Identificable ({value_cls}).")
                value_id = int(value)
            elif isinstance(value, str):
                try:
                    value_id = int(value)
                    if value_id < 0:
                        raise ValueError()
                except ValueError:
                    raise ValueError(f"Valor inválido: {value}")
            elif isinstance(value, int):
                value_id = value
            else:
                raise TypeError(f"Valor inválido: {value} (tipo {type(value)})")

            result[key_id] = value_id

        if key_cls is None or value_cls is None:
            raise ValueError("No se pudo inferir key_cls o value_cls a partir de los elementos.")

        # Validación de que todos los ids de llaves y valores están en el registro.
        for key_id in result:
            if key_id not in key_cls.get_registro():
                raise ValueError(f"{key_id} no está presente en el registro de la clase {key_cls}")
        for value_id in result.values():
            if value_id not in value_cls.get_registro():
                raise ValueError(f"{value_id} no está presente en el registro de la clase {value_cls}")

        self.key_cls = key_cls
        self.value_cls = value_cls
        super().__init__(result)

    def __setitem__(self, key: int | str | K, value: int | str | V) -> None:
        key_id = validate_id(key, self.key_cls)
        value_id = validate_id(value, self.value_cls)
        super().__setitem__(key_id, value_id)

    def keys_ids(self) -> set[int]:
        return set(super().keys())

    def values_ids(self) -> set[int]:
        return set(super().values())

    def keys(self) -> IdSet[K]:
        return IdSet(self.keys_ids(), cls=self.key_cls)

    def values(self) -> IdSet[V]:
        return IdSet(self.values_ids(), cls=self.value_cls)

    def keys_objects(self) -> set[K]:
        return {self.key_cls.from_id(id_) for id_ in self.keys_ids()}

    def values_objects(self) -> set[V]:
        return {self.value_cls.from_id(id_) for id_ in self.values_ids()}

    def items(self) -> set[tuple[int, int]]:
        return set(super().items())

    def items_objects(self) -> set[tuple[K, V]]:
        return {
            (self.key_cls.from_id(k_id), self.value_cls.from_id(v_id))
            for k_id, v_id in self.items()
        }


class IdSet(set[int], Generic[T]):

    cls: type[T] | None

    def __init__(
        self: IdSet[T],
        items: Iterable[int | str | T] = (),
        cls: type[T] | None = None
    ) -> None:
        if cls is not None and not issubclass(cls, Identificable):
            raise ValueError(f"No se puede crear una lista de IDs de tipo {cls} porque no extiende de Identificable.")

        self.cls = cls
        set_ids: set[int] = {}

        for item in items:

            # Si item es Identificable, se infiere de él el tipo de la lista si no se había inferido anteriormente,
            # y en otro caso, si el tipo del item no coincide con el tipo inferido, se lanza un error.
            if isinstance(item, Identificable):
                if self.cls is None:
                    self.cls = type(item)
                elif not isinstance(item, self.cls):
                    raise ValueError(f"Todos los objetos deben ser del mismo tipo Identificable ({self.cls}).")
                set_ids.add(int(item))

            # Si item es un entero, se comprueba si es positivo.
            elif isinstance(item, int):
                if item < 0:
                    raise ValueError(f"ID inválido {item}: debe ser un entero positivo.")
                set_ids.add(item)

            # Si item es una cadena, se comprueba si se puede convertir a un entero positivo.
            elif isinstance(item, str):
                try:
                    val = int(item)
                    if val < 0:
                        raise ValueError()
                    set_ids.add(val)
                except ValueError:
                    raise ValueError(f"El string '{item}' no representa un ID entero válido.")

            # Si item no es un Identificable, int o str, se lanza un error.
            else:
                raise TypeError(f"Elemento inválido en el conjunto: {item} (tipo {type(item)})")

        # Si self.cls es None, no se pudo inferir la clase de la IdList tras iterar los elementos del conjunto
        # recibida como parámetro, luego se lanza un error, pues no se puede dar tipo al objeto IdList.
        if self.cls is None:
            raise ValueError("No se pudo inferir el tipo cls a partir de los elementos proporcionados.")

        # Si de algún modo se infirió un tipo que no extiende de Identificable, se lanza un error.
        if not issubclass(self.cls, Identificable):
            raise ValueError(f"El tipo {self.cls} no es una subclase válida de Identificable.")

        # Se itera de nuevo en el conjunto comprobando que todos los ids guardados aparecen en el registro del tipo
        # del conjunto. Si alguno no aparece, se lanza un error.
        registro = self.cls.get_registro()
        for id_ in set_ids:
            if id_ not in registro:
                raise ValueError(f"{id_} no está presente en el registro de la clase {self.cls}")

        super().__init__(set_ids)

    def __contains__(self: IdSet[T], item: int | str | T) -> bool:
        try:
            id_ = validate_id(item, self.cls)
        except ValueError:
            return False
        return id_ in self

    def objects(self: IdSet[T]) -> set[T]:
        return {self.cls.from_id(id_) for id_ in self}

    def add(self: IdSet[T], element: int | str | T) -> None:
        super().add(validate_id(element, self.cls))

    def remove(self: IdSet[T], element: int | str | T) -> None:
        super().remove(validate_id(element, self.cls))

    def filter(self: IdSet[T], predicate: Callable[[T], bool]) -> IdSet[T]:
        return IdSet({int(obj) for obj in self.objects() if predicate(obj)}, cls=self.cls)

    def map(self: IdSet[T], f: Callable[[T], Any]) -> set[Any]:
        return {f(obj) for obj in self.objects()}

    def map_to_identificable(self: IdSet[T], f: Callable[[T], Identificable]) -> IdSet[T]:
        resultado = {f(obj) for obj in self.objects()}
        if not resultado:
            raise ValueError("No se pudieron mapear los objetos, la lista resultante está vacía.")
        cls_inferida = type(next(iter(resultado)))
        if not all(isinstance(obj, cls_inferida) for obj in resultado):
            raise ValueError("Todos los objetos mapeados deben ser del mismo tipo Identificable.")
        return IdSet({int(obj) for obj in resultado}, cls=cls_inferida)

    def flatmap(self: IdSet[T], f: Callable[[T], Iterable[Any]]) -> set[Any]:
        return {valor for obj in self.objects() for valor in f(obj)}

    def flatmap_to_identificable(self: IdSet[T], f: Callable[[T], Iterable[Identificable]]) -> IdSet[T]:
        resultado = {valor for obj in self.objects() for valor in f(obj)}
        if not resultado:
            raise ValueError("No se pudieron flatmapear los objetos, la lista resultante está vacía.")
        cls_inferida = type(next(iter(resultado)))
        if not all(isinstance(r, cls_inferida) for r in resultado):
            raise ValueError("Todos los objetos mapeados deben ser del mismo tipo Identificable.")
        return IdSet({int(obj) for obj in resultado}, cls=cls_inferida)

    def any_match(self: IdSet[T], predicate: Callable[[T], bool]) -> bool:
        return any(predicate(obj) for obj in self.objects())

    def all_match(self: IdSet[T], predicate: Callable[[T], bool]) -> bool:
        return all(predicate(obj) for obj in self.objects())

    def none_match(self: IdSet[T], predicate: Callable[[T], bool]) -> bool:
        return not any(predicate(obj) for obj in self.objects())

    def for_each(self: IdSet[T], consumer: Callable[[T], None]) -> None:
        for obj in self.objects():
            consumer(obj)

    def union(self: IdSet[T], other: IdSet[T]) -> IdSet[T]:
        if not isinstance(other, IdSet):
            raise TypeError("La operación unión solo es válida con otro IdSet.")
        if self.cls != other.cls:
            raise ValueError("No se puede hacer la unión de dos IdSet con clases distintas.")
        return IdSet(super().union(other), cls=self.cls)

    def intersection(self: IdSet[T], other: IdSet[T]) -> IdSet[T]:
        if not isinstance(other, IdSet):
            raise TypeError("La operación intersección solo es válida con otro IdSet.")
        if self.cls != other.cls:
            raise ValueError("No se puede hacer la intersección de dos IdSet con clases distintas.")
        return IdSet(super().intersection(other), cls=self.cls)

    def difference(self: IdSet[T], other: IdSet[T]) -> IdSet[T]:
        if not isinstance(other, IdSet):
            raise TypeError("La operación diferencia solo es válida con otro IdSet.")
        if self.cls != other.cls:
            raise ValueError("No se puede hacer la diferencia de dos IdSet con clases distintas.")
        return IdSet(super().difference(other), cls=self.cls)

    def symmetric_difference(self: IdSet[T], other: IdSet[T]) -> IdSet[T]:
        if not isinstance(other, IdSet):
            raise TypeError("La operación diferencia simétrica solo es válida con otro IdSet.")
        if self.cls != other.cls:
            raise ValueError("No se puede hacer la diferencia simétrica de dos IdSet con clases distintas.")
        return IdSet(super().symmetric_difference(other), cls=self.cls)

    def is_subset_of(self: IdSet[T], other: IdSet[T]) -> bool:
        if not isinstance(other, IdSet):
            raise TypeError("La operación subconjunto solo es válida con otro IdSet.")
        if self.cls != other.cls:
            raise ValueError("No se pueden comparar conjuntos con clases distintas.")
        return self.issubset(other)

    def is_superset_of(self: IdSet[T], other: IdSet[T]) -> bool:
        if not isinstance(other, IdSet):
            raise TypeError("La operación superconjunto solo es válida con otro IdSet.")
        if self.cls != other.cls:
            raise ValueError("No se pueden comparar conjuntos con clases distintas.")
        return self.issuperset(other)


class IdList(list[int], Generic[T]):
    """
    Clase que extiende de list[int], listando los ids de una cierta clase Identificable, guardada en un atributo cls, y
    que en la inicialización será dado explícitamente o deducido del iterable recibido como argumento.

    Iterar sobre un objeto IdList iterará sobre los ids enteros. Si se quiere acceder una la lista que contiene los
    objetos de Identificable que representan, se usará objects().

    Los métodos __contains__, __setitem__, __add__, __mul__, append y remove fueron sobreescritos para que acepten como
    argumento objetos de Identificable, así como int o str (que represente un entero acorde) para mayor flexibilidad.
    """

    cls: type[T] | None

    def __init__(
        self: IdList[T],
        items: Iterable[int | str | T] = (),
        cls: type[T] | None = None
    ) -> None:
        if cls is not None and not issubclass(cls, Identificable):
            raise ValueError(f"No se puede crear una lista de IDs de tipo {cls} porque no extiende de Identificable.")

        self.cls = cls
        lista_ids: list[int] = []

        for item in items:

            # Si item es Identificable, se infiere de él el tipo de la lista si no se había inferido anteriormente,
            # y en otro caso, si el tipo del item no coincide con el tipo inferido, se lanza un error.
            if isinstance(item, Identificable):
                if self.cls is None:
                    self.cls = type(item)
                elif not isinstance(item, self.cls):
                    raise ValueError(f"Todos los objetos deben ser del mismo tipo Identificable ({self.cls}).")
                lista_ids.append(int(item))

            # Si item es un entero, se comprueba si es positivo.
            elif isinstance(item, int):
                if item < 0:
                    raise ValueError(f"ID inválido {item}: debe ser un entero positivo.")
                lista_ids.append(item)

            # Si item es una cadena, se comprueba si se puede convertir a un entero positivo.
            elif isinstance(item, str):
                try:
                    val = int(item)
                    if val < 0:
                        raise ValueError()
                    lista_ids.append(val)
                except ValueError:
                    raise ValueError(f"El string '{item}' no representa un ID entero válido.")

            # Si item no es un Identificable, int o str, se lanza un error.
            else:
                raise TypeError(f"Elemento inválido en la lista: {item} (tipo {type(item)})")

        # Si self.cls es None, no se pudo inferir la clase de la IdList tras iterar los elementos de la lista
        # recibida como parámetro, luego se lanza un error, pues no se puede dar tipo al objeto IdList.
        if self.cls is None:
            raise ValueError("No se pudo inferir el tipo cls a partir de los elementos proporcionados.")

        # Si de algún modo se infirió un tipo que no extiende de Identificable, se lanza un error.
        if not issubclass(self.cls, Identificable):
            raise ValueError(f"El tipo {self.cls} no es una subclase válida de Identificable.")

        # Se itera de nuevo en la lista comprobando que todos los ids guardados aparecen en el registro del tipo
        # de la lista. Si alguno no aparece, se lanza un error.
        registro = self.cls.get_registro()
        for id_ in lista_ids:
            if id_ not in registro:
                raise ValueError(f"{id_} no está presente en el registro de la clase {self.cls}")

        super().__init__(lista_ids)

    def __getitem__(self: IdList[T], index: int | slice) -> T | IdList[T]:
        if isinstance(index, slice):
            return IdList(self[index.start:index.stop:index.step], cls=self.cls)
        return super().__getitem__(index)

    def __setitem__(self: IdList[T], index: int, value: int | str | T) -> None:
        try:
            id_ = validate_id(value, self.cls)
            super().__setitem__(index, id_)
        except ValueError:
            raise ValueError(f"{value} no es un valor válido.")

    def __contains__(self: IdList[T], item: int | str | T) -> bool:
        try:
            id_ = validate_id(item, self.cls)
        except ValueError:
            return False
        return id_ in self

    def __add__(self: IdList[T], other: Any) -> IdList[T]:
        if not isinstance(other, IdList):
            raise TypeError("Solo se puede sumar otro IdList.")
        if self.cls != other.cls:
            raise ValueError("Ambas listas deben tener el mismo tipo cls para poder sumarse.")
        return IdList(list(self) + list(other), cls=self.cls)

    def __iadd__(self: IdList[T], other: Any) -> IdList[T]:
        if not isinstance(other, IdList):
            raise TypeError("Solo se puede usar '+=' con otro IdList.")
        if self.cls != other.cls:
            raise ValueError("Ambas listas deben tener el mismo tipo cls para poder sumarse.")
        self.extend(other)
        return self

    def __mul__(self: IdList[T], value: int) -> IdList[T]:
        if not isinstance(value, int):
            raise TypeError("El valor de multiplicación debe ser un entero.")
        return IdList(super().__mul__(value), cls=self.cls)

    def __rmul__(self: IdList[T], value: int) -> IdList[T]:
        return self.__mul__(value)

    def objects(self: IdList[T]) -> list[T]:
        return [self.cls.from_id(id_) for id_ in self]

    def extend(self: IdList[T], items: Iterable[int | str | T]) -> None:
        for item in items:
            self.append(item)

    def get_object(self: IdList[T], index: int, fallback: T | None = None) -> T | None:
        try:
            id_ = self[index]
            return self.cls.from_id(id_) or fallback
        except IndexError:
            return fallback

    def append(self: IdList[T], element: int | str | T) -> None:
        super().append(validate_id(element, self.cls))

    def remove(self: IdList[T], element: int | str | T) -> None:
        super().remove(validate_id(element, self.cls))

    def insert(self: IdList[T], index: int, element: int | str | T) -> None:
        super().insert(index, validate_id(element, self.cls))

    def sort(self: IdList[T], key: Callable[[T], Any] = lambda x: x.id, reverse: bool = False) -> None:
        objects = self.objects()
        objects.sort(key=key, reverse=reverse)
        self[:] = [int(obj) for obj in objects]

    def sorted(self: IdList[T], key: Callable[[T], Any] = lambda x: x.id, reverse: bool = False) -> IdList[T]:
        objects = self.objects()
        objects.sort(key=key, reverse=reverse)
        return IdList(objects, cls=self.cls)

    def filter(self: IdList[T], predicate: Callable[[T], bool]) -> IdList[T]:
        return IdList([int(obj) for obj in self.objects() if predicate(obj)], cls=self.cls)

    def map(self: IdList[T], f: Callable[[T], Any]) -> list[Any]:
        return [f(obj) for obj in self.objects()]

    def map_to_identificable(self: IdList[T], f: Callable[[T], Identificable]) -> IdList[T]:
        resultado = [f(obj) for obj in self.objects()]
        if not resultado:
            raise ValueError("No se pudieron mapear los objetos, la lista resultante está vacía.")
        cls_inferida = type(resultado[0])
        if not all(isinstance(obj, cls_inferida) for obj in resultado):
            raise ValueError("Todos los objetos mapeados deben ser del mismo tipo Identificable.")
        return IdList([int(obj) for obj in resultado], cls=cls_inferida)

    def flatmap(self: IdList[T], f: Callable[[T], Iterable[Any]]) -> list[Any]:
        return [valor for obj in self.objects() for valor in f(obj)]

    def flatmap_to_identificable(self: IdList[T], f: Callable[[T], Iterable[Identificable]]) -> IdList[T]:
        resultado = [valor for obj in self.objects() for valor in f(obj)]
        if not resultado:
            raise ValueError("No se pudieron flatmapear los objetos, la lista resultante está vacía.")
        cls_inferida = type(resultado[0])
        if not all(isinstance(r, cls_inferida) for r in resultado):
            raise ValueError("Todos los objetos mapeados deben ser del mismo tipo Identificable.")
        return IdList([int(obj) for obj in resultado], cls=cls_inferida)

    def any_match(self: IdList[T], predicate: Callable[[T], bool]) -> bool:
        return any(predicate(obj) for obj in self.objects())

    def all_match(self: IdList[T], predicate: Callable[[T], bool]) -> bool:
        return all(predicate(obj) for obj in self.objects())

    def none_match(self: IdList[T], predicate: Callable[[T], bool]) -> bool:
        return not any(predicate(obj) for obj in self.objects())

    def distinct(self: IdList[T]) -> IdList[T]:
        ids_vistos = set()
        ids_unicos = []
        for id_ in self:
            if id_ not in ids_vistos:
                ids_vistos.add(id_)
                ids_unicos.append(id_)
        return IdList(ids_unicos, self.cls)

    def for_each(self: IdList[T], consumer: Callable[[T], None]) -> None:
        for obj in self.objects():
            consumer(obj)


class IdTuple(tuple[int, ...]):
    classes: tuple[type[Identificable], ...]

    def __new__(
        cls,
        items: Iterable[int | Identificable],
        classes: tuple[type[Identificable], ...]
    ) -> IdTuple:
        items_list = list(items)

        if len(items_list) != len(classes):
            raise ValueError("El número de elementos no coincide con el número de clases.")

        validated_ids: list[int] = []

        for index, (item, expected_cls) in enumerate(zip(items_list, classes)):
            if not issubclass(expected_cls, Identificable):
                raise TypeError(f"La clase en la posición {index} no extiende Identificable.")

            if isinstance(item, Identificable):
                if not isinstance(item, expected_cls):
                    raise TypeError(
                        f"El objeto en la posición {index} debe ser una instancia de {expected_cls}, no de {type(item)}."
                    )
                validated_ids.append(int(item))
            elif isinstance(item, int):
                if item < 0:
                    raise ValueError(f"ID inválido en la posición {index}: {item}")
                validated_ids.append(item)
            else:
                raise TypeError(
                    f"Elemento inválido en la posición {index}: se esperaba int o Identificable, se recibió {type(item)}."
                )

        obj = super().__new__(cls, tuple(validated_ids))
        obj.classes = classes
        return obj

    def objects(self: IdTuple) -> tuple[Identificable, ...]:
        return tuple(cls.from_id(id_) for id_, cls in zip(self, self.classes))

    def get_object(self: IdTuple, index: int) -> T:
        return self.classes[index].from_id(self[index])


@dataclass(slots=True, frozen=True)
class IdKey(Generic[T]):
    """
    Clase que representa un objeto de una clase que extienda Identificable con la información mínima requerida del
    mismo: su ID y su tipo, que son guardados como atributos de la clase IdKey.

    En la inicialización de IdKey se puede recibir un ID representado por un int o str, en cuyo caso será obligado
    dar la clase de forma explícita, o recibir un objeto Identificable, del cual se podrá inferir la clase, pero si
    esta es dada explícitamente, deberán coincidir.

    Mediante get_objet() se obtendrá el objeto Identificable representado por la IdKey.
    """

    id: int
    cls: type[T]

    def __init__(self: IdKey[T], item: int | str | T, cls: type[T] | None = None) -> None:
        if isinstance(item, Identificable):
            if cls is not None and not isinstance(item, cls):
                raise ValueError(f"El objeto {item} no es de tipo {cls}")
            cls = cls or type(item)
        else:
            if cls is None:
                raise ValueError("cls debe ser proporcionado si item no es Identificable")

        id_ = validate_id(item, cls)

        object.__setattr__(self, "id", id_)
        object.__setattr__(self, "cls", cls)

    def __eq__(self: IdKey[T], other: object) -> bool:
        if isinstance(other, Identificable):
            return isinstance(other, self.cls) and self.id == int(other)
        if isinstance(other, IdKey):
            return self.cls == other.cls and self.id == other.id
        return False

    def __hash__(self: IdKey[T]) -> int:
        return hash((self.id, self.cls))

    def get_object(self: IdKey[T]) -> T:
        return self.cls.from_id(self.id)


@dataclass(eq=False, slots=True, frozen=True)
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


@dataclass(eq=False, slots=True, frozen=True)
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


@dataclass(eq=False, slots=True, frozen=True)
class Contrato(Identificable):
    nombre_es: int

    def __post_init__(self: Contrato) -> None:
        super(Contrato, self).__post_init__()


@dataclass(eq=False, slots=True, frozen=True)
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


def validate_id(element: int | str | T, cls: type[Identificable]) -> int:
    if isinstance(element, Identificable):
        # Si element es Identificable, se comprueba que sea de la clase parámetro.
        if not isinstance(element, cls):
            raise ValueError(f"El objeto {element} no es de tipo {cls}")
        id_ = int(element)

    elif isinstance(element, int):
        # Si element es un entero, se comprueba que sea mayor que cero.
        if element < 0:
            raise ValueError(f"ID inválido {element}: debe ser un entero positivo.")
        id_ = element

    elif isinstance(element, str):
        # Si element es una cadena, se comprueba que representa un entero mayor que cero.
        try:
            id_ = int(element)
            if id_ < 0:
                raise ValueError()
        except ValueError:
            raise ValueError(f"El string '{element}' no representa un ID entero válido.")

    else:
        # Si element no es Identificable, int o str, se lanza un error.
        raise TypeError(f"Error al validar {element}: es de tipo {type(element)}")

    if id_ not in cls.get_registro():
        # Finalmente, se comprueba que el ID entero obtenido está en el registro de la clase.
        raise ValueError(f"El id {id_} no está en el registro de {cls}")

    return id_


def parse_bool(value: str | int) -> bool:
    return str(value).strip() in {'true', 'True', '1', 'yes', 'Yes', 'y', 'Y'}


if __name__ == "__main__":
    t1: Trabajador = Trabajador(id=1, nombre="Alfonso", apellidos="Gallego", codigo=5000)
    t2: Trabajador = Trabajador(id=2, nombre="Pepito", apellidos="Pérez", codigo=5001)
    t3: Trabajador = Trabajador(id=3, nombre="María", apellidos="Gómez", codigo=5002)
    t4: Trabajador = Trabajador(id=4, nombre="José Manuel", apellidos="Fernández", codigo=5003)
    t5: Trabajador = Trabajador(id=5, nombre="Ana", apellidos="Díaz", codigo=5004)

    lista_de_ids: IdList = IdList([t1, t2, t3])
    lista_de_ids.insert(1, t4)
    lista_de_ids.insert(3, t5)
    print(lista_de_ids)
    lista_de_ids.for_each(lambda t: print(t.id, t.nombre, t.apellidos))
    print("------------------------------------------------------")
    lista_de_ids.filter(lambda t: t.nombre.startswith("A")).for_each(lambda t: print(t.id, t.nombre, t.apellidos))
    print("------------------------------------------------------")
    print(lista_de_ids.map(lambda t: str(t.id) + " - " + str(t.codigo)))



