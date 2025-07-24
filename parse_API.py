from typing import TypeVar, Any, NamedTuple, Literal, Callable
from collections import defaultdict, namedtuple
from itertools import product
import json

import requests

from ClasesMetodosAuxiliares import DatosTrabajadoresPuestosJornadas, ListasPreferencias
from Clases import Trabajador, PuestoTrabajo, NivelDesempeno, Jornada, TipoJornada, parse_bool

get_id = lambda p : p.id
get_codigo = lambda t : t.codigo


url_puestos: Callable[[int], str]
url_demandas: Callable[[int], str]
url_concesiones: Callable[[int], str]
url_excepciones: Callable[[int], str]
url_eventos: Callable[[int], str]
url_jornadas: str
url_grupos: Callable[[int], str]
url_contratos: Callable[[int], str]


def fetch_data_from_api(api_url: str) -> dict[str, Any] | None:
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except requests.exceptions.ConnectionError as conn_err:
        print(f"Connection error occurred: {conn_err}")
    except requests.exceptions.Timeout as timeout_err:
        print(f"Timeout error: {timeout_err}")
    except requests.exceptions.RequestException as req_err:
        print(f"Unexpected error: {req_err}")
    return None



def setup_json_url(
    server: str,
    fecha: str,
    usuario_origen_autenticacion_clave: str,
    usuario_alias: str,
    usuario_password: str
):
    # TODO: Ocultar parámetros usuario_origen_autenticacion_clave, usuario_alias, usuario_password en las url
    global url_puestos, url_demandas, url_concesiones, url_excepciones, url_eventos, url_jornadas, concesiones_data, url_grupos, url_contratos

    token_autentificacion: str = (
        f'?usuario_origen_autenticacion_clave={usuario_origen_autenticacion_clave}'
        f'&usuario_alias={usuario_alias}'
        f'&usuario_password={usuario_password}'
    )

    url_puestos = lambda page : (
        f'{server}/ws/contractual/trabajadores_puestos_trabajos/listar/'
        f'condicionesFechas:%7B"fecha":%20"{fecha}"%7D/'
        f'limit:1000/page:{page}'
        + token_autentificacion
    )

    url_demandas = lambda page : (
        f'{server}/ws/contratacion/demandas/listar/'
        f'contiene:["DemandasPuestosTrabajos"]/'
        f'condicionesFechas:%7B"fecha":%20"{fecha}"%7D/'
        f'limit:1000/page:{page}'
        +token_autentificacion
    )

    url_concesiones = lambda page : (
        f'{server}/ws/disponibilidad/tipos_concesiones_trabajadores/listar'
        f'/condicionesFechas:%7B"fecha":%20"{fecha}"%7D/'
        f'limit:1000/page:{page}/'
        + token_autentificacion
    )

    url_excepciones = lambda page : (
        f'{server}/ws/disponibilidad/tipos_excepciones_trabajadores/listar/'
        f'condicionesFechas:%7B"fecha":%20"{fecha}"%7D/'
        f'limit:1000/page:{page}/'
        + token_autentificacion
    )

    url_eventos = lambda page : (
        f'{server}/ws/disponibilidad/eventos_trabajadores_grupos_personales/listar/'
        f'condicionesFechas:%7B"fecha":%20"{fecha}"%7D/'
        f'limit:1000/page:{page}'
        + token_autentificacion
    )

    url_jornadas = f'{server}/ws/operativa/jornadas/listar/' + token_autentificacion

    url_grupos = lambda page : (
        f'{server}/ws/contractual/trabajadores_grupos_personales/listar/'
        f'condicionesFechas:%7B"fecha":%20"{fecha}"%7D/'
        f'limit:1000/page:{page}'
        + token_autentificacion
    )

    url_contratos = lambda page : (
        f'{server}/ws/contractual/trabajadores_contratos/listar/'
        f'condicionesFechas:%7B"fecha":%20"{fecha}"%7D/'
        f'limit:1000/page:{page}'
        + token_autentificacion
    )


def parse_trabajadores_puestos() -> dict[PuestoTrabajo, list[Trabajador]]:
    # puesto -> lista ordenada de trabajadores especialistas en ese puesto.
    dict_especialidades: dict[PuestoTrabajo, list[Trabajador]] = defaultdict(list)

    pagina = 1
    puestos_data = fetch_data_from_api(url_puestos(pagina))
    while puestos_data:
        for entry in puestos_data:  # type: dict[str, Any]
            # Si ya se parseó un trabajador, puesto o nivel, se obtiene de un registro interno, si no se crea uno nuevo.
            trabajador: Trabajador = Trabajador.get_or_create(entry["Trabajador"])
            puesto: PuestoTrabajo = PuestoTrabajo.get_or_create(entry["PuestoTrabajo"])
            nivel: NivelDesempeno = NivelDesempeno.get_or_create(entry["NivelDesempeno"])
            # Se modifica el atributo capacidades del trabajador para reflejar su eficacia en el puesto de trabajo.
            trabajador.actualizar_capacidades(puesto, nivel)
            # Si el ID del NivelDesempeno es 1, es una especialidad, así que se actualiza la lista apropiada.
            if nivel.id == 1:
                dict_especialidades[puesto].append(trabajador)
        pagina += 1
        puestos_data = fetch_data_from_api(url_puestos(pagina))

    for lista_especialidades in dict_especialidades.values(): # type: list[Trabajador]
        lista_especialidades.sort(key=get_codigo)
    return dict_especialidades


def parse_demandas() -> dict[tuple[PuestoTrabajo, Jornada], int]:
    demandas: dict[tuple[PuestoTrabajo, Jornada], int] = defaultdict(lambda:0)
    pagina = 1
    demandas_data = fetch_data_from_api(url_demandas(pagina))
    while demandas_data:
        for entry in demandas_data:  # type: dict[str, Any]
            if entry["DemandasPuestosTrabajos"]:
                jornada = Jornada.from_id(int(entry["Demanda"]["jornada_id"]))
                for demanda_puestos in entry["DemandasPuestosTrabajos"]:  # type: dict[str, Any]
                    puesto = PuestoTrabajo.from_id(demanda_puestos["puesto_trabajo_id"])
                    demandas[puesto, jornada] += int(demanda_puestos["num_trabajadores"])
        pagina += 1
        demandas_data = fetch_data_from_api(url_demandas(pagina))
    return demandas


def parse_excepciones() -> set[tuple[Trabajador, Jornada]]:
    set_excepciones_jornadas: set[tuple[Trabajador, Jornada]] = set()
    set_excepciones_dias: set[Trabajador] = set()

    pagina = 1
    excepciones_data = fetch_data_from_api(url_excepciones(pagina))

    while excepciones_data:
        for entry in excepciones_data:  # type: dict[str, Any]
            if entry.get("TiposExcepcionesTrabajadoresJornadas", {}):
                trabajador = Trabajador.from_id(entry["TipoExcepcionTrabajador"]["trabajador_id"])
                for excepcion in entry["TiposExcepcionesTrabajadoresJornadas"]:  # type: dict[str, Any]
                    set_excepciones_jornadas.add((trabajador, Jornada.from_id(excepcion["jornada_id"])))
        pagina += 1
        excepciones_data = fetch_data_from_api(url_excepciones(pagina))

    pagina = 1
    eventos_data = fetch_data_from_api(url_eventos(pagina))

    while eventos_data:
        for entry in eventos_data:  # type: dict[str, Any]
            if not parse_bool(entry["Evento"]["disponibilidad"]):
                set_excepciones_dias.add(Trabajador.from_id(entry["EventoTrabajadorGrupoPersonal"]["trabajador_id"]))
        pagina += 1
        eventos_data = fetch_data_from_api(url_eventos(pagina))

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

    pagina = 1
    concesiones_data = fetch_data_from_api(url_concesiones(pagina))

    while concesiones_data:
        for entry in concesiones_data:  # type: dict[str, Any]
            if entry["TipoConcesion"]["nombre_es"] == "Voluntario Noche" or entry["TipoConcesion"]["nombre_es"] == "Voluntario Doble":
                trabajador = Trabajador.from_id(entry["TipoConcesionTrabajador"]["trabajador_id"])
                if entry["TipoConcesion"]["nombre_es"] == "Voluntario Noche" and trabajador is not None and trabajador not in lista_voluntarios_noche:
                    lista_voluntarios_noche.append(trabajador)
                if entry["TipoConcesion"]["nombre_es"] == "Voluntario Doble" and trabajador is not None and trabajador not in lista_voluntarios_doble:
                    lista_voluntarios_doble.append(trabajador)
        pagina += 1
        concesiones_data = fetch_data_from_api(url_concesiones(pagina))

    lista_voluntarios_doble.sort(key=get_codigo)
    lista_voluntarios_noche.sort(key=get_codigo)

    return lista_voluntarios_doble, lista_voluntarios_noche


Grupo = Literal["Grupo1", "Grupo2", "Grupo3", "Grupo4"]
def parse_grupos(nombre_grupo_tarde: Grupo):
    grupo_manana: list[Trabajador] = []
    grupo_tarde: list[Trabajador] = []

    pagina = 1
    grupos_data = fetch_data_from_api(url_grupos(pagina))

    while grupos_data:
        for entry in grupos_data:  # type: dict[str, Any]
            if entry["GrupoPersonal"]["nombre_es"] == nombre_grupo_tarde:
                grupo_tarde.append(Trabajador.get_or_create(entry["Trabajador"]))
            else:
                grupo_manana.append(Trabajador.get_or_create(entry["Trabajador"]))
        pagina += 1
        grupos_data = fetch_data_from_api(url_grupos(pagina))

    grupo_manana.sort(key=get_codigo)
    grupo_tarde.sort(key=get_codigo)

    return grupo_manana, grupo_tarde


def parse_contratos():
    lista_trabajadores: list[Trabajador] = []

    pagina = 1
    contratos_data = fetch_data_from_api(url_contratos(pagina))

    while contratos_data:
        for entry in contratos_data:  # type: dict[str, Any]
            trabajador: Trabajador = Trabajador.from_id(entry["TrabajadorContrato"]["trabajador_id"])
            if trabajador is not None:
                lista_trabajadores.append(trabajador)
        pagina += 1
        contratos_data = fetch_data_from_api(url_contratos(pagina))

    return lista_trabajadores


def parse_all_data(server: str, fecha: str, nombre_grupo_tarde: Grupo) -> tuple[
    DatosTrabajadoresPuestosJornadas,           # Lista de Trabajadores, Puestos y Jornadas
    ListasPreferencias,                         # Listas de preferencias de especialidad, voluntarios de noche y dobles, mañana y tarde
    dict[tuple[PuestoTrabajo, Jornada], int],   # Demanda por cada puesto y jornada
    set[tuple[Trabajador, Jornada]],            # Disponibilidad de los trabajadores en las distintas jornadas
]:
    setup_json_url(server, fecha, usuario_origen_autenticacion_clave, usuario_alias, usuario_password)

    especialidades = parse_trabajadores_puestos()
    demandas = parse_demandas()
    disponibilidad = parse_excepciones()
    voluntarios_doble, voluntarios_noche = parse_concesiones()
    preferencia_manana, preferencia_tarde = parse_grupos(nombre_grupo_tarde)
    trabajadores = parse_contratos()
    puestos: list[PuestoTrabajo] = list(PuestoTrabajo.get_registro().values())
    jornadas: list[Jornada] = list(Jornada)

    return (
        DatosTrabajadoresPuestosJornadas(trabajadores, puestos, jornadas),
        ListasPreferencias(especialidades, {TipoJornada.MANANA : preferencia_manana, TipoJornada.TARDE : preferencia_tarde, TipoJornada.NOCHE : voluntarios_noche}, voluntarios_doble),
        demandas,
        disponibilidad
    )

"""
[sensitive data here]
"""

data = parse_all_data(bilbao, dia, "Grupo1")
