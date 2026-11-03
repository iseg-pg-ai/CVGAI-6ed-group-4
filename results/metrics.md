# Métricas do benchmark

- **Motor OCR:** easyocr
- **Método de extração:** llm_local
- **Pré-processamento:** ativo
- **Documentos avaliados:** 27

## Global

| Métrica | Valor |
| --- | --- |
| taxa_acerto | 0.3778 |
| taxa_falso_positivo | 0.1667 |
| taxa_omissao | 0.2093 |

## Por campo

| Dimensão | taxa_acerto | taxa_falso_positivo | taxa_omissao |
| --- | --- | --- | --- |
| entidade | 0.1481 | 0.0000 | 0.0000 |
| nif_comerciante | 0.1852 | 1.0000 | 0.1923 |
| nif_cliente | 0.2963 | 0.0000 | 0.7727 |
| data | 0.7407 | 0.0000 | 0.0741 |
| total | 0.5185 | 0.0000 | 0.1111 |

## Por grupo

| Dimensão | taxa_acerto | taxa_falso_positivo | taxa_omissao |
| --- | --- | --- | --- |
| direito | 0.3625 | 0.2500 | 0.2368 |
| amarrotado | 0.4000 | 0.0000 | 0.1698 |
