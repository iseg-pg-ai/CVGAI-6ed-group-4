"""Prompts and JSON schemas for LLM-based structured receipt extraction.

Ref: Docs/implementation_plan.md, section 7, WP-10.
"""

from __future__ import annotations

from typing import Any

from src.core.types import FIELD_NAMES
from src.extraction.llm_client import build_object_schema, parse_json_object

#: System prompt.
#:
#: Written in Portuguese because the corpus and the end users are Portuguese. It describes
#: the JSON shape in prose on purpose: ``llama-server`` ignores ``response_format`` when
#: building the prompt, so a model without the shape in text invents its own key names.
SYSTEM_PROMPT = """\
És um extrator de dados de talões e faturas simplificadas portuguesas.

Recebes o texto de um talão, linha a linha, numerado no formato [NN]. Preenche os cinco
campos pedidos usando APENAS o que está escrito nesse texto.

Regras obrigatórias:
1. Não inventes. Se um campo não estiver no documento, devolve "" nesse campo e false na
   chave correspondente de "presente". Se o campo estiver no documento, devolve o valor e true na chave correspondente de "presente".
2. Não calcules nada. Não somes artigos, não derives subtotais, não estimes valores e não
   convertas quantidades em preços. O total só existe se estiver impresso numa linha com a
   palavra TOTAL. Um talão que só tem linhas de artigos tem total "" e false.
3. Em "linhas" indica, para cada campo que devolvas como presente, o número NN da linha onde
   esse valor está impresso. Um campo presente tem sempre um número de linha maior que 0.
   Usa 0 apenas para os campos ausentes.
4. nif_comerciante é o NIF da empresa ou loja emitente (em Portugal começa por 5). nif_cliente é o NIF
   do cliente ou adquirente (em Portugal começa por 1, 2 ou 3, ou surge junto a Cliente, Nome, Contrib ou N.C.).
   O código 999999990 significa consumidor final: devolve "" e false em nif_cliente.
5. Os NIF têm exatamente 9 dígitos, sem espaços. Nunca uses números de telefone ou de apoio ao cliente (como números iniciados por 21, 22, 808, 800) nem códigos postais (como 1495-158 ou 2790-237) como NIF. Se o número impresso não tiver 9 dígitos, copia-o tal como está.
6. "data" no formato AAAA-MM-DD. Se só houver uma data de emissão, é essa.
7. "total" é o valor numérico efetivamente pago com IVA incluído, com ponto decimal e sem símbolo de moeda.
   Se houver "Total a Pagar", usa esse valor. Ignora subtotais, trocos e valores entregues.
8. "entidade" é a designação comercial ou nome da loja emitente, sem morada nem telefone.

Responde apenas com um objeto JSON, sem texto à volta e sem blocos de código, com exatamente
estas chaves:

{
  "entidade": "<string>",
  "nif_comerciante": "<string>",
  "nif_cliente": "<string>",
  "data": "<AAAA-MM-DD>",
  "total": "<decimal>",
  "presente": {
    "entidade": <true|false>,
    "nif_comerciante": <true|false>,
    "nif_cliente": <true|false>,
    "data": <true|false>,
    "total": <true|false>
  },
  "linhas": {
    "entidade": <inteiro>,
    "nif_comerciante": <inteiro>,
    "nif_cliente": <inteiro>,
    "data": <inteiro>,
    "total": <inteiro>
  }
}"""


#: Structured-output schema sent as ``response_format`` or ``response_schema``.
SCHEMA: dict[str, Any] = build_object_schema(
    {
        **{name: {"type": "string"} for name in FIELD_NAMES},
        "presente": build_object_schema({name: {"type": "boolean"} for name in FIELD_NAMES}),
        "linhas": build_object_schema({name: {"type": "integer"} for name in FIELD_NAMES}),
    }
)


class LLMResponseError(RuntimeError):
    """Raised when the model output cannot be turned into an extraction."""


def parse_response_payload(raw: str) -> dict[str, Any]:
    """Recover the JSON object from a model response."""
    return parse_json_object(raw, LLMResponseError)


_parse_json_object = parse_response_payload
