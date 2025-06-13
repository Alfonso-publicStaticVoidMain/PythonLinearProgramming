from __future__ import annotations
import re
from collections import defaultdict
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict, PositiveInt
from typing import TypeVar, ClassVar, Type, Any, Iterable

T = TypeVar("T", bound="Identificable")


class Identificable(BaseModel):
    id: PositiveInt

    _registro: ClassVar[dict[type[Identificable], dict[int, Identificable]]] = defaultdict(dict)

    model_config = ConfigDict(extra='ignore', frozen=True, arbitrary_types_allowed=True)

    def __init__(self, **data):
        if type(self) is Identificable:
            raise TypeError("Identificable es una clase abstracta, que no puede ser instanciada.")
        super().__init__(**data)

    def model_post_init(self: Identificable, __context: Any) -> None:
        registro: dict[int, Identificable] = type(self).get_registro()
        if self.id not in registro:
            registro[self.id] = self

    @classmethod
    def from_id(cls: type[T], obj_id: PositiveInt | str) -> T | None:
        try:
            obj_id = int(obj_id)
        except (ValueError, TypeError):
            return None
        return cls.get_registro().get(obj_id, None)

    @classmethod
    def get_registro(cls: type[T]) -> dict[int, T]:
        return cls._registro.get(cls, {})

    @classmethod
    def get_or_create(cls: Type[T], data: dict[str, Any]) -> T | None:
        instance = cls.from_id(data.get("id"))
        return instance or cls.model_validate(data)

    @classmethod
    def get_by_id(cls: Type[T], collection: Iterable[T], id_: PositiveInt | str, fallback: T | None = None) -> T | None:
        """
        Busca en un iterable de tipo T (con atributo id: int) el único objeto con un cierto ID.
        :param collection: Iterable de tipo T, el cual debe tener implementado un atributo id: int.
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

    def __str__(self):
        return f"{type(self).__name__}(id={self.id})"

    def __repr__(self):
        return self.__str__()

    def __format__(self: Identificable, format_spec: str) -> str:
        return format(str(self), format_spec)


class Trabajador(Identificable):
    nombre: str
    apellidos: str
    especialidades: set[PuestoTrabajo] = Field(default_factory=set)
    capacidades: dict[PuestoTrabajo, NivelDesempeno] = Field(default_factory=dict)

    def __post_init__(self: Trabajador) -> None:
        super(Trabajador, self).__post_init__()

    def actualizar_capacidades(self: Trabajador, puesto: PuestoTrabajo, nivel: NivelDesempeno) -> None:
        self.capacidades[puesto] = nivel
        if nivel.id == 1: self.especialidades.add(puesto)

    @field_validator("nombre", "apellidos")
    def validate_name(cls: Any, v: str) -> str:
        if not re.fullmatch(r"[A-Za-zÁÉÍÓÚáéíóúÑñÜü' -]+", v):
            raise ValueError("Invalid characters in name")
        return v

    def __str__(self: Trabajador) -> str:
        return f'{self.id}'

    def __repr__(self: Trabajador) -> str:
        return f'Trabajador {self.id} - {self.nombre} {self.apellidos}'


class TrabajadorPuestoTrabajo(Identificable):
    trabajador: Trabajador
    puesto_trabajo: PuestoTrabajo
    nivel_desempeno: NivelDesempeno

    def model_post_init(self: TrabajadorPuestoTrabajo, __context: Any) -> None:
        super().model_post_init(__context)

    @property
    def get_trabajador_id(self: TrabajadorPuestoTrabajo) -> PositiveInt:
        return self.trabajador.id

    @property
    def get_trabajador_nombre_y_apellidos(self: TrabajadorPuestoTrabajo) -> str:
        return self.trabajador.nombre + " " + self.trabajador.apellidos

    @property
    def get_puesto_trabajo_id(self: TrabajadorPuestoTrabajo) -> PositiveInt:
        return self.puesto_trabajo.id

    @property
    def get_puesto_trabajo_nombre(self: TrabajadorPuestoTrabajo) -> str:
        return self.puesto_trabajo.nombre_es

    @property
    def nivel_desempeno_id(self: TrabajadorPuestoTrabajo) -> PositiveInt:
        return self.nivel_desempeno.id

    @property
    def nivel_desempeno_nombre(self: TrabajadorPuestoTrabajo) -> str:
        return self.nivel_desempeno.nombre_es


class PuestoTrabajo(Identificable):
    nombre_es: str

    def model_post_init(self: PuestoTrabajo, __context: Any) -> None:
        super().model_post_init(__context)

    def __str__(self: PuestoTrabajo) -> str:
        return f'id={self.id} {self.nombre_es}'

    def __repr__(self: PuestoTrabajo) -> str:
        return f'PuestoTrabajo(id={self.id} {self.nombre_es})'

    def __format__(self: PuestoTrabajo, format_spec: str) -> str:
        return format(str(self), format_spec)


class NivelDesempeno(Identificable):
    nombre_es: str

    def model_post_init(self: NivelDesempeno, __context: Any) -> None:
        super().model_post_init(__context)

    def __str__(self: NivelDesempeno) -> str:
        return f'{self.id}: {self.nombre_es}'

    def __repr__(self: NivelDesempeno) -> str:
        return f'NivelDesempeno({self.id}, {self.nombre_es})'

    def __format__(self: NivelDesempeno, format_spec: str) -> str:
        return format(str(self), format_spec)