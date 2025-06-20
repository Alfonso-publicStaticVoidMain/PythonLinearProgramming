from typing import Type, TypeVar, Any, get_type_hints
from collections import defaultdict
from itertools import product
import json

from Clases import Trabajador, PuestoTrabajo, NivelDesempeno, Jornada, parse_bool, parse_data_into
from asignacion import realizar_asignacion

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


# Se cargan los archivos JSON conteniendo los datos de interés, que se guardan en un diccionario cada uno.
puestos_data: dict[str, Any] = load_json_file("../data/trabajadore_puestos.json")
demandas_data: dict[str, Any] = load_json_file("../data/demandas.json")
voluntarios_data: dict[str, Any] = load_json_file("../data/concesiones.json")
excepciones_data: dict[str, Any] = load_json_file("../data/excepciones_trabajadores.json")
eventos_data: dict[str, Any] = load_json_file("../data/eventos_trabajadores.json")
jornadas_data: dict[str, Any] = load_json_file("../data/jornadas.json")
concesiones_data: dict[str, Any] = load_json_file("../data/concesiones.json")
grupos_data: dict[str, Any] = load_json_file("../data/trabajadores_grupos_personales.json")
contratos_data: dict[str, Any] = load_json_file("../data/contratos.json")


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
    demandas: dict[tuple[PuestoTrabajo, Jornada], int] = {}
    for entry in demandas_data: # type: dict[str, Any]
        if entry["DemandasPuestosTrabajos"]:
            jornada = Jornada.from_id(int(entry["Demanda"]["jornada_id"]))
            for demanda_puestos in entry["DemandasPuestosTrabajos"]: # type: dict[str, Any]
                puesto = PuestoTrabajo.from_id(demanda_puestos["puesto_trabajo_id"])
                if (puesto, jornada) in demandas:
                    demandas[puesto, jornada] += int(demanda_puestos["num_trabajadores"])
                else:
                    demandas[puesto, jornada] = int(demanda_puestos["num_trabajadores"])
    return demandas


def parse_excepciones() -> set[tuple[Trabajador, Jornada]]:
    set_excepciones_jornadas: set[tuple[Trabajador, Jornada]] = set()
    set_excepciones_dias: set[Trabajador] = set()
    for entry in excepciones_data: # type: dict[str, Any]
        if entry["TiposExcepcionesTrabajadoresJornadas"]:
            trabajador = Trabajador.from_id(entry["TipoExcepcionTrabajador"]["trabajador_id"])
            for excepcion in entry["TiposExcepcionesTrabajadoresJornadas"]: # type: dict[str, Any]
                set_excepciones_jornadas.add((trabajador, Jornada.from_id(excepcion["jornada_id"])))

    for entry in eventos_data: # type: dict[str, Any]
        if not parse_bool(entry["Evento"]["disponibilidad"]):
            set_excepciones_dias.add(Trabajador.from_id(entry["EventoTrabajadorGrupoPersonal"]["trabajador_id"]))

    return {
        (trabajador, jornada)
        for trabajador, jornada in product(Trabajador.get_registro().values(), Jornada) # type: Trabajador, Jornada
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
            if entry["TipoConcesion"]["nombre_es"] == "Voluntario Noche":
                lista_voluntarios_noche.append(trabajador)
            if entry["TipoConcesion"]["nombre_es"] == "Voluntario Doble":
                lista_voluntarios_doble.append(trabajador)
    lista_voluntarios_doble.sort(key=get_codigo)
    lista_voluntarios_noche.sort(key=get_codigo)
    return lista_voluntarios_doble, lista_voluntarios_noche


def parse_grupo1_2():
    grupo1: list[Trabajador] = []
    grupo2: list[Trabajador] = []
    for entry in grupos_data: # type: dict[str, Any]
        if entry["GrupoPersonal"]["nombre_es"] == "Grupo1":
            grupo1.append(Trabajador.get_or_create(entry["Trabajador"]))
        elif entry["GrupoPersonal"]["nombre_es"] == "Grupo2":
            grupo2.append(Trabajador.get_or_create(entry["Trabajador"]))
    grupo1.sort(key=get_codigo)
    grupo2.sort(key=get_codigo)
    return grupo1, grupo2


def parse_contratos():
    lista_trabajadores: list[Trabajador] = []
    for entry in contratos_data: # type: dict[str, Any]
        lista_trabajadores.append(Trabajador.from_id(entry["TrabajadorContrato"]["trabajador_id"]))
    return lista_trabajadores


def parse_all_data() -> tuple[
    list[Trabajador],       # Lista de trabajadores
    list[PuestoTrabajo],    # Lista de puestos
    list[Jornada],          # Lista de jornadas
    dict[tuple[PuestoTrabajo, Jornada], int],   # Demanda por cada puesto y jornada
    dict[PuestoTrabajo, list[Trabajador]],      # Lista de trabajadores con cada puesto como especialidad
    list[Trabajador],       # Voluntarios de noche
    set[tuple[Trabajador, Jornada]],            # Disponibilidad de los trabajadores en las distintas jornadas
    list[Trabajador],       # Voluntarios para dobles
    list[Trabajador],       # Preferencia de mañana
    list[Trabajador]        # Preferencia de tarde
]:
    # No tocar el orden de los métodos llamados dentro de este método, o todo puede romperse.
    # Si, se que es mal diseño que ocurra eso, pero es lo que hay.

    # Al ejecutarse el siguiente método se guardan en el registro los trabajadores y puestos que se encuentren en el
    # JSON trabajadore_puestos.json. Por lo tanto las sentencias que utilicen el registro de estas clases deberán
    # ejecutarse después, o lo encontrarán vacío.
    especialidades: dict[PuestoTrabajo, list[Trabajador]] = parse_trabajadores_puestos()
    # Deben estar después de parse_trabajadores_puestos
    trabajadores: list[Trabajador] = parse_contratos() # list(Trabajador.get_registro().values())
    puestos: list[PuestoTrabajo] = list(PuestoTrabajo.get_registro().values())
    jornadas: list[Jornada] = list(Jornada)
    # parse_demandas accede al registro de PuestoTrabajo
    demandas: dict[tuple[PuestoTrabajo, Jornada], int] = parse_demandas()
    # parse_excepciones accede al registro de Trabajador
    disponibilidad = parse_excepciones()
    voluntarios_doble, voluntarios_noche = parse_concesiones()

    # Intercambiar el orden de asignación de estas dos listas cambiará que grupo tiene preferencia de mañana o de tarde.
    # Por defecto, parse_grupo1_2 retorna grupo1, grupo2.
    preferencia_manana, preferencia_tarde = parse_grupo1_2()

    return trabajadores, puestos, jornadas, demandas, especialidades, voluntarios_noche, disponibilidad, voluntarios_doble, preferencia_manana, preferencia_tarde


data = parse_all_data()


if __name__ == "__main__":
    solucion = realizar_asignacion(
        *data,
        verbose_estadisticas_avanzadas=True,
        verbose_general=True,
        verbose_asignacion_trabajadores=True,
        verbose_asignacion_puestos=False,
        forzar_maximo_asignaciones_especialidad=False
    )