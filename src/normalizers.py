"""Funções  para normalizar e comparar valores de campos de recibos."""

from __future__ import annotations

import re
import unicodedata
from datetime import date
from decimal import Decimal, InvalidOperation

VALID_NIF_FIRST_DIGITS = frozenset("12356789")


def limpar_texto(raw: str) -> str:
    """Remove espaços em branco, elimina caracteres de controlo e corta as extremidades.

    Argumentos:

    raw: Texto a limpar.

    Retorna: Texto sem caracteres de controlo e com espaços simples entre as palavras.
    """
    characters: list[str] = []
    for character in raw:
        if character.isspace():
            characters.append(" ")
        elif unicodedata.category(character) != "Cc":
            characters.append(character)
    return " ".join("".join(characters).split())


def parse_moeda(raw: str) -> Decimal | None:
    """Analisa o primeiro valor monetário interpretável no texto.

    As vírgulas são tratadas como separadores decimais. Um ponto seguido de exatamente três
    dígitos é tratado como separador de milhares, assim como espaços agrupados.

    Argumentos:

    raw: Texto que pode conter um valor monetário e um marcador de moeda.

    Retorna:

    O valor analisado ou ``None`` se não estiver presente nenhum valor válido.
    """
    match = re.search(r"(?<!\d)[+-]?\d+(?:[ \u00a0\u202f.,]\d+)*(?!\d)", limpar_texto(raw))
    if match is None:
        return None

    token = match.group().replace(" ", "").replace("\u00a0", "").replace("\u202f", "")
    comma_index = token.rfind(",")
    dot_index = token.rfind(".")
    decimal_separator: str | None = None

    if comma_index >= 0 and dot_index >= 0:
        decimal_separator = "," if comma_index > dot_index else "."
    elif comma_index >= 0:
        decimal_separator = ","
    elif dot_index >= 0:
        digits_after_dot = len(token) - dot_index - 1
        if digits_after_dot != 3:
            decimal_separator = "."

    if decimal_separator is None:
        normalized = token.replace(",", "").replace(".", "")
    else:
        decimal_index = token.rfind(decimal_separator)
        integer_part = token[:decimal_index].replace(",", "").replace(".", "")
        fractional_part = token[decimal_index + 1 :]
        normalized = f"{integer_part}.{fractional_part}"

    try:
        return Decimal(normalized)
    except InvalidOperation:
        return None


def parse_data(raw: str) -> date | None:
    """Analisa datas de recibos comuns, opcionalmente seguidas de um horário.

    Anos com dois dígitos em datas com hífen são interpretados como ``YY-MM-DD``.
    Com quatro dígitos, aceita ``YYYY-MM-DD`` ou ``DD-MM-YYYY``. Com barra ou ponto,
    aceita ``YYYY/MM/DD`` ou ``DD/MM/YY`` (e os formatos equivalentes com ponto).

    Argumentos:

    raw: Texto que contém uma data.

    Retorna:

    Uma data analisada ou ``None`` quando a entrada não é uma data válida e suportada.
    """
    text = limpar_texto(raw)
    text = re.sub(r"(?<!\d)\d{1,2}:\d{2}(?::\d{2})?(?!\d)", "", text)
    text = limpar_texto(text)
    date_patterns = (
        (r"(?<!\d)(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})(?!\d)", "ymd"),
        (r"(?<!\d)(\d{2})-(\d{1,2})-(\d{1,2})(?!\d)", "ymd"),
        (r"(?<!\d)(\d{8})(?!\d)", "compact"),
        (r"(?<!\d)(\d{1,2})[-/.](\d{1,2})[-/.](\d{2}|\d{4})(?!\d)", "dmy"),
    )

    for pattern, order in date_patterns:
        match = re.search(pattern, text)
        if match is None:
            continue

        if order == "compact":
            compact = match.group(1)
            year, month, day = int(compact[:4]), int(compact[4:6]), int(compact[6:])
        elif order == "ymd":
            first, second, third = (int(part) for part in match.groups())
            year, month, day = first, second, third
            if year < 100:
                year += 2000
        else:
            first, second, third = (int(part) for part in match.groups())
            day, month, year = first, second, third
            if year < 100:
                year += 2000

        try:
            return date(year, month, day)
        except ValueError:
            return None

    return None


def parse_nif(raw: str) -> str | None:
    """Extrai um código NIF português de nove dígitos a partir de um texto.

    Argumentos:

    raw: Texto que pode conter um código NIF, com espaços facultativos entre os dígitos.

    Retorna:

    Os nove dígitos, ou ``None`` se não estiver presente nenhuma sequência de nove dígitos.
    """
    match = re.search(r"(?<!\d)\d(?:[ \t]*\d){8}(?!\d)", raw)
    if match is None:
        return None
    return "".join(match.group().split())


def validar_nif(nif: str) -> bool:
    """Valida o primeiro dígito permitido e o dígito verificador módulo 11 de um NIF português.

    Argumentos:

    nif: NIF candidato.

    Retorno:

    `True`` se o NIF tiver nove dígitos e um dígito de controlo válido.
    """
    if len(nif) != 9 or not nif.isdigit() or nif[0] not in VALID_NIF_FIRST_DIGITS:
        return False

    checksum = sum(int(digit) * weight for digit, weight in zip(nif[:8], range(9, 1, -1), strict=True))
    check_digit = 11 - checksum % 11
    if check_digit >= 10:
        check_digit = 0
    return int(nif[-1]) == check_digit


def normalizar_entidade(raw: str) -> str:
    """Cria um nome de entidade canónico que ignora os acentos e a pontuação.

    Argumentos:

    raw: Nome da entidade tal como é lido de um recibo.

    Retorno:

    Nome em minúsculas, sem acentos, pontuação ou sufixos legais reconhecidos.
    """
    decomposed = unicodedata.normalize("NFD", raw.lower())
    entity_suffixes = {"lda", "sa", "unipessoal"}
    without_accents = "".join(character for character in decomposed if unicodedata.category(character) != "Mn")
    without_periods = without_accents.replace(".", "")
    words = "".join(
        character if character.isalnum() or character.isspace() else " " for character in without_periods
    ).split()
    while words and words[-1] in entity_suffixes:
        words.pop()
    return " ".join(words)


def valores_equivalentes(campo: str, esperado: str | None, obtido: str | None) -> bool:
    """Compara dois valores utilizando a regra de normalização para o seu campo canónico.

    Argumentos:

    campo: Nome do campo canónico (``total``, ``data``, um campo ``nif_*`` ou
    ``entidade``).

    esperado: Valor de referência.

    adquirido: Valor extraído.

    Retorna:

    Se ambos os valores são equivalentes de acordo com a regra do campo.

    Exceções:

    ValueError: Se ``campo`` não for um dos campos canónicos suportados.
    """
    esperado = esperado.strip() or None if esperado is not None else None
    obtido = obtido.strip() or None if obtido is not None else None

    if esperado is None or obtido is None:
        return esperado is None and obtido is None

    if campo == "total":
        expected_value = parse_moeda(esperado)
        obtained_value = parse_moeda(obtido)
    elif campo == "data":
        expected_value = parse_data(esperado)
        obtained_value = parse_data(obtido)
    elif campo in {"nif_comerciante", "nif_cliente"}:
        expected_value = parse_nif(esperado)
        obtained_value = parse_nif(obtido)
    elif campo == "entidade":
        expected_value = normalizar_entidade(esperado)
        obtained_value = normalizar_entidade(obtido)
    else:
        raise ValueError(f"Unsupported field: {campo}")

    return expected_value is not None and obtained_value is not None and expected_value == obtained_value
