from typing import TypeVar, Any, NamedTuple, Literal, Callable
from collections import defaultdict, namedtuple
from itertools import product
from functional import seq
import json

from ClasesMetodosAuxiliares import DatosTrabajadoresPuestosJornadas, ListasPreferencias, \
    IdsTrabajadoresPuestosJornadas, IdsListasPreferencias
from Clases import Trabajador, PuestoTrabajo, NivelDesempeno, Jornada, TipoJornada, IdList, parse_bool, IdDict

T = TypeVar('T')
get_id = lambda p : p.id
get_codigo = lambda t : t.codigo


def load_json_file(path: str) -> dict[str, Any] | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"Invalid JSON in {path}: {e}")
    except FileNotFoundError as e:
        print(f"JSON file not found: {e}")
    return None


data: str = " " + "2025-05-05"
# Se cargan los archivos JSON conteniendo los datos de interés, que se guardan en un diccionario cada uno.
puestos_data: dict[str, Any] = load_json_file("../data" + data + "/trabajadore_puestos.json")
demandas_data: dict[str, Any] = load_json_file("../data" + data + "/demandas.json")
voluntarios_data: dict[str, Any] = load_json_file("../data" + data + "/concesiones.json")
excepciones_data: dict[str, Any] = load_json_file("../data" + data + "/excepciones_trabajadores.json")
eventos_data: dict[str, Any] = load_json_file("../data" + data + "/eventos_trabajadores.json")
jornadas_data: dict[str, Any] = load_json_file("../data" + data + "/jornadas.json")
tipos_jornadas_data: dict[str, Any] = load_json_file("../data" + data + "/tipos_jornadas.json")
concesiones_data: dict[str, Any] = load_json_file("../data" + data + "/concesiones.json")
grupos_data: dict[str, Any] = load_json_file("../data" + data + "/trabajadores_grupos_personales.json")
contratos_data: dict[str, Any] = load_json_file("../data" + data + "/contratos.json")


def ordena_ids_trabajadores(lista_trabajadores: list[int], key: Callable[[Trabajador], int] = get_codigo) -> list[int]:
    """
    Recibe como argumento una lista de IDs que representan trabajadores. Pasa esa lista a la lista de los objetos de
    Trabajador que representan, los ordena por la llave argumento (por defecto por su código), los devuelve a sus IDs,
    y retorna la lista resultante.
    """
    if not lista_trabajadores:
        return []
    return (
        seq(lista_trabajadores)
        .map(lambda id_: Trabajador.from_id(id_))
        .sorted(key=key)
        .map(int)
        .to_list()
    )


def parse_trabajadores_puestos_ids() ->dict[int, list[int]]:
    """
    Retorna un diccionario que mapea: id_puesto -> lista de ids de los trabajadores que son especialistas en ese puesto.
    """
    especialistas_por_puesto: dict[int, list[int]] = defaultdict(list)
    for entry in puestos_data: # type: dict[str, Any]
        # Se parsea la información del JSON en un objeto de Trabajador, PuestoTrabajo y NivelDesempeno
        trabajador: Trabajador = Trabajador.get_or_create(entry["Trabajador"])
        puesto: PuestoTrabajo = PuestoTrabajo.get_or_create(entry["PuestoTrabajo"])
        nivel: NivelDesempeno = NivelDesempeno.get_or_create(entry["NivelDesempeno"])
        # Se modifica el atributo capacidades del trabajador para reflejar su eficacia en el puesto de trabajo, y
        # a continuación también se modifica el diccionario global a retornar que engloba a todos los trabajadores.
        trabajador.actualizar_capacidades(puesto, nivel)
        # Si el ID del NivelDesempeno es 1, es una especialidad, así que se actualizan las listas apropiadas.
        if nivel.id == 1:
            especialistas_por_puesto[puesto.id].append(trabajador.id)
    # Se ordena cada lista por el código de los trabajadores. Para alcanzar esto, primero hay que convertir los IDs en
    # objetos de Trabajador mediante from_id, ordenarlos por su código, luego volverlos a convertir en un ID con int().
    for ids_trabajadores_especialistas in especialistas_por_puesto.values():
        ids_trabajadores_especialistas = ordena_ids_trabajadores(ids_trabajadores_especialistas)
    return especialistas_por_puesto


def parse_jornadas_ids() -> list[int]:
    """
    Retorna la lista de todas las jornadas.
    """
    jornadas: list[int] = []

    # Recorre el json de tipos de jornadas y crea un objeto para cada uno, almacenándolo automáticamente en el registro
    # de Identificable para su posterior acceso mediante from_id.
    for entry in tipos_jornadas_data["rows"]: # type: dict[str, Any]
        TipoJornada.get_or_create(entry)

    # Recorre el json de jornadas y crea un objeto para cada una. Automáticamente, también le asigna un atributo
    # tipo_jornada que referencia al objeto de TipoJornada apropiado, gracias a que estos fueron registrados al crearse.
    for entry in jornadas_data["rows"]:
        jornadas.append(Jornada.get_or_create(entry).id)

    return jornadas


def parse_demandas_ids() -> dict[tuple[int, int], int]:
    """
    Retorna un diccionario que mapea: (puesto_id, jornada_id) -> cantidad de trabajadores demandada para ese puesto en
    esa jornada.
    """
    demandas_por_puesto_y_jornada: dict[tuple[int, int], int] = defaultdict(lambda:0)
    for entry in demandas_data: # type: dict[str, Any]
        if entry["DemandasPuestosTrabajos"]:
            jornada_id = entry["Demanda"]["jornada_id"]
            for demanda_puestos in entry["DemandasPuestosTrabajos"]: # type: dict[str, Any]
                puesto_id = demanda_puestos["puesto_trabajo_id"]
                demandas_por_puesto_y_jornada[puesto_id, jornada_id] += int(demanda_puestos["num_trabajadores"])

    return demandas_por_puesto_y_jornada


def parse_excepciones_ids() -> set[tuple[int, int]]:
    """
    Retorna un conjunto de tuplas (trabajador_id, jornada_id) en las que el trabajador está disponible en la jornada.
    """
    excep_trab_jorn: set[tuple[int, int]] = set()
    excepciones_trabajadores: set[int] = set()

    for entry in excepciones_data: # type: dict[str, Any]
        if entry.get("TiposExcepcionesTrabajadoresJornadas", None):
            trabajador_id: int = int(entry["TipoExcepcionTrabajador"]["trabajador_id"])
            for excepcion in entry["TiposExcepcionesTrabajadoresJornadas"]: # type: dict[str, Any]
                jornada_id: int = int(excepcion["jornada_id"])
                excep_trab_jorn.add((trabajador_id, jornada_id))

    for entry in eventos_data: # type: dict[str, Any]
        if not parse_bool(entry["Evento"]["disponibilidad"]):
            excepciones_trabajadores.add(entry["EventoTrabajadorGrupoPersonal"]["trabajador_id"])

    # Se retornan los pares (trabajador_id, jornada_id) que no aparezcan en excep_trab_jorn y cuyo ID del trabajador no
    # aparezca en excepciones_trabajadores. Para ello se deben tener previamente registrado todos los trabajadores y
    # jornadas en el registro de Identificable (es decir, se deben haber creado los objetos de Trabajador y Jornada)
    return {
        (trabajador_id, jornada_id)
        for trabajador_id, jornada_id in product(Trabajador.get_registro().keys(), Jornada.get_registro().keys())
        if (trabajador_id, jornada_id) not in excep_trab_jorn
        and trabajador_id not in excepciones_trabajadores
    }


def parse_concesiones_ids() -> tuple[
    list[int],   # Trabajadores voluntarios dobles
    list[int]    # Trabajadores voluntarios de noche
]:
    """
    Retorna dos listas de IDs de trabajadores: los voluntarios para dobles y para noche, ordenados por el código de los
    trabajadores que representan.
    """
    voluntarios_doble: list[int] = []
    voluntarios_noche: list[int] = []
    for entry in concesiones_data: # type: dict[str, Any]
        tipo_voluntario = entry["TipoConcesion"]["nombre_es"]
        if tipo_voluntario == "Voluntario Noche" or tipo_voluntario == "Voluntario Doble":
            trabajador_id = int(entry["TipoConcesionTrabajador"]["trabajador_id"])
            if tipo_voluntario == "Voluntario Noche" and trabajador_id not in voluntarios_noche:
                voluntarios_noche.append(trabajador_id)
            if tipo_voluntario == "Voluntario Doble" and trabajador_id not in voluntarios_doble:
                voluntarios_doble.append(trabajador_id)

    return ordena_ids_trabajadores(voluntarios_doble), ordena_ids_trabajadores(voluntarios_noche)


Grupo = Literal["Grupo1", "Grupo2", "Grupo3", "Grupo4"]
def parse_grupos_ids(nombre_grupo_tarde: Grupo) -> tuple[
    list[int],  # Trabajadores con preferencia de mañana.
    list[int]   # Trabajadores con preferencia de tarde.
]:
    """
    Retorna dos listas de IDs de trabajadores: los de preferencia de mañana y los de tarde, ordenados por el código
    de los trabajadores que representan.

    Acepta como parámetro una cadena de entre ["Grupo1", "Grupo2", "Grupo3", "Grupo4"], representando el nombre del
    grupo que tendrá preferencia de tarde.
    """
    grupo_manana: list[int] = []
    grupo_tarde: list[int] = []
    for entry in grupos_data: # type: dict[str, Any]
        if entry["GrupoPersonal"]["nombre_es"] == nombre_grupo_tarde:
            grupo_tarde.append(int(entry["Trabajador"]["id"]))
        else:
            grupo_manana.append(int(entry["Trabajador"]["id"]))

    return ordena_ids_trabajadores(grupo_manana), ordena_ids_trabajadores(grupo_tarde)


def parse_contratos_ids() -> list[int]:
    """
    Devuelve la lista de trabajadores con contratos activos, que son potencialmente elegibles para las asignaciones.
    """
    lista_trabajadores: list[int] = []
    ids_trabajadores: set[int] = set(Trabajador.get_registro().keys())
    for entry in contratos_data: # type: dict[str, Any]
        trabajador_id = int(entry["TrabajadorContrato"]["trabajador_id"])
        if trabajador_id in ids_trabajadores:
            lista_trabajadores.append(trabajador_id)
    return lista_trabajadores


def parse_all_data_ids(nombre_grupo_tarde: Grupo) -> tuple[
    IdsTrabajadoresPuestosJornadas,     # Lista de Trabajadores, Puestos y Jornadas
    IdsListasPreferencias,              # Listas de preferencias de especialidad, voluntarios de noche y dobles, mañana y tarde
    dict[tuple[int, int], int],         # Demanda por cada puesto y jornada
    set[tuple[int, int]],               # Disponibilidad de los trabajadores en las distintas jornadas
]:
    jornadas: list[int] = parse_jornadas_ids()
    especialidades: dict[int, list[int]] = parse_trabajadores_puestos_ids()
    trabajadores: list[int] = parse_contratos_ids()
    puestos: list[int] = list(PuestoTrabajo.get_registro().keys())
    demandas: dict[tuple[int, int], int] = parse_demandas_ids()
    disponibilidad = parse_excepciones_ids()
    voluntarios_doble, voluntarios_noche = parse_concesiones_ids()
    preferencia_manana, preferencia_tarde = parse_grupos_ids(nombre_grupo_tarde)
    return (
        IdsTrabajadoresPuestosJornadas(trabajadores, puestos, jornadas),
        IdsListasPreferencias(especialidades, {1 : preferencia_manana, 2 : preferencia_tarde, 3 : voluntarios_noche}, voluntarios_doble),
        demandas,
        disponibilidad
    )


data_ids = parse_all_data_ids("Grupo1")