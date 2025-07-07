from typing import TypeVar, Any, NamedTuple, Literal
from collections import defaultdict, namedtuple
from itertools import product
import json

from ClasesMetodosAuxiliares import DatosTrabajadoresPuestosJornadas, ListasPreferencias
from Clases import Trabajador, PuestoTrabajo, NivelDesempeno, Jornada, parse_bool, TipoJornada

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
concesiones_data: dict[str, Any] = load_json_file("../data" + data + "/concesiones.json")
grupos_data: dict[str, Any] = load_json_file("../data" + data + "/trabajadores_grupos_personales.json")
contratos_data: dict[str, Any] = load_json_file("../data" + data + "/contratos.json")


def parse_trabajadores_puestos() -> dict[PuestoTrabajo, list[Trabajador]]:
    # Se van a devolver un diccionario indexado por tuplas (trabajador, puesto), mapeándolas a un objeto NivelDesempeno,
    # que representa la eficacia del trabajador en ese puesto, siendo un ID de 1 su especialidad principal, y un
    # diccionario indexado por PuestoTrabajo que mapea a una lista de los trabajadores que tienen esa especialidad.
    dict_especialidades: dict[PuestoTrabajo, list[Trabajador]] = defaultdict(list)
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
            dict_especialidades[puesto].append(trabajador)
    for lista_especialidades in dict_especialidades.values(): # type: list[Trabajador]
        lista_especialidades.sort(key=get_codigo)
    return dict_especialidades


def parse_demandas() -> dict[tuple[PuestoTrabajo, Jornada], int]:
    demandas: dict[tuple[PuestoTrabajo, Jornada], int] = defaultdict(lambda:0)
    for entry in demandas_data: # type: dict[str, Any]
        if entry["DemandasPuestosTrabajos"]:
            jornada = Jornada.from_id(int(entry["Demanda"]["jornada_id"]))
            for demanda_puestos in entry["DemandasPuestosTrabajos"]: # type: dict[str, Any]
                puesto = PuestoTrabajo.from_id(demanda_puestos["puesto_trabajo_id"])
                demandas[puesto, jornada] += int(demanda_puestos["num_trabajadores"])
    return demandas


def parse_excepciones() -> set[tuple[Trabajador, Jornada]]:
    set_excepciones_jornadas: set[tuple[Trabajador, Jornada]] = set()
    set_excepciones_dias: set[Trabajador] = set()

    for entry in excepciones_data: # type: dict[str, Any]
        if entry.get("TiposExcepcionesTrabajadoresJornadas", {}):
            trabajador = Trabajador.from_id(entry["TipoExcepcionTrabajador"]["trabajador_id"])
            for excepcion in entry["TiposExcepcionesTrabajadoresJornadas"]: # type: dict[str, Any]
                set_excepciones_jornadas.add((trabajador, Jornada.from_id(excepcion["jornada_id"])))

    for entry in eventos_data: # type: dict[str, Any]
        if not parse_bool(entry["Evento"]["disponibilidad"]):
            set_excepciones_dias.add(Trabajador.from_id(entry["EventoTrabajadorGrupoPersonal"]["trabajador_id"]))

    return {
        (trabajador, jornada)
        for trabajador, jornada in product(Trabajador.get_registro().values(), Jornada)
        if (trabajador, jornada) not in set_excepciones_jornadas
        and trabajador not in set_excepciones_dias
    }


def parse_concesiones() -> tuple[
    list[Trabajador],   # Voluntarios dobles
    list[Trabajador]    # Voluntarios de noche
]:
    lista_voluntarios_doble: list[Trabajador] = []
    lista_voluntarios_noche: list[Trabajador] = []
    for entry in concesiones_data: # type: dict[str, Any]
        if entry["TipoConcesion"]["nombre_es"] == "Voluntario Noche" or entry["TipoConcesion"]["nombre_es"] == "Voluntario Doble":
            trabajador = Trabajador.from_id(entry["TipoConcesionTrabajador"]["trabajador_id"])
            if entry["TipoConcesion"]["nombre_es"] == "Voluntario Noche" and trabajador is not None and trabajador not in lista_voluntarios_noche:
                lista_voluntarios_noche.append(trabajador)
            if entry["TipoConcesion"]["nombre_es"] == "Voluntario Doble" and trabajador is not None and trabajador not in lista_voluntarios_doble:
                lista_voluntarios_doble.append(trabajador)
    lista_voluntarios_doble.sort(key=get_codigo)
    lista_voluntarios_noche.sort(key=get_codigo)
    return lista_voluntarios_doble, lista_voluntarios_noche


Grupo = Literal["Grupo1", "Grupo2", "Grupo3", "Grupo4"]
def parse_grupos(nombre_grupo_tarde: Grupo):
    grupo_manana: list[Trabajador] = []
    grupo_tarde: list[Trabajador] = []
    for entry in grupos_data: # type: dict[str, Any]
        if entry["GrupoPersonal"]["nombre_es"] == nombre_grupo_tarde:
            grupo_tarde.append(Trabajador.get_or_create(entry["Trabajador"]))
        else:
            grupo_manana.append(Trabajador.get_or_create(entry["Trabajador"]))
    grupo_manana.sort(key=get_codigo)
    grupo_tarde.sort(key=get_codigo)
    return grupo_manana, grupo_tarde


def parse_contratos():
    lista_trabajadores: list[Trabajador] = []
    for entry in contratos_data: # type: dict[str, Any]
        trabajador = Trabajador.from_id(entry["TrabajadorContrato"]["trabajador_id"])
        if trabajador is not None:
            lista_trabajadores.append(trabajador)
    return lista_trabajadores


def parse_all_data(nombre_grupo_tarde: Grupo) -> tuple[
    DatosTrabajadoresPuestosJornadas,           # Lista de Trabajadores, Puestos y Jornadas
    ListasPreferencias,                         # Listas de preferencias de especialidad, voluntarios de noche y dobles, mañana y tarde
    dict[tuple[PuestoTrabajo, Jornada], int],   # Demanda por cada puesto y jornada
    set[tuple[Trabajador, Jornada]],            # Disponibilidad de los trabajadores en las distintas jornadas
]:
    especialidades: dict[PuestoTrabajo, list[Trabajador]] = parse_trabajadores_puestos()
    trabajadores: list[Trabajador] = parse_contratos()
    puestos: list[PuestoTrabajo] = list(PuestoTrabajo.get_registro().values())
    jornadas: list[Jornada] = list(Jornada)
    demandas: dict[tuple[PuestoTrabajo, Jornada], int] = parse_demandas()
    disponibilidad = parse_excepciones()
    voluntarios_doble, voluntarios_noche = parse_concesiones()
    preferencia_manana, preferencia_tarde = parse_grupos(nombre_grupo_tarde)
    return (
        DatosTrabajadoresPuestosJornadas(trabajadores, puestos, jornadas),
        ListasPreferencias(especialidades, {TipoJornada.MANANA : preferencia_manana, TipoJornada.TARDE : preferencia_tarde, TipoJornada.NOCHE : voluntarios_noche}, voluntarios_doble),
        demandas,
        disponibilidad
    )


data = parse_all_data("Grupo1")