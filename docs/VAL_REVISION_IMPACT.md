# Impacto da revisão do VAL nos códigos publicados

**Cleilson Elias de Sousa · Auditoria de 11 de novembro de 2026**

## Conclusão e fontes

A revisão torna VAL um controlador testável de admissão e recuperação de memória
por domínio. Ela não invalida automaticamente LLI-CAMS ou CALMS-SAVANT: nenhum
deles importa VAL ou depende do antigo teorema para executar. Corrige, porém,
o alcance das interpretações científicas e impede transferir garantias entre
mecanismos que apenas compartilham nomenclatura.

Foram inspecionados código, testes, exemplos e documentos formais dos dois
repositórios públicos encontrados na conta do autor. Os arquivos recuperados
foram conferidos contra seus hashes de blob Git.

| Projeto | Estado anterior inspecionado |
|---|---|
| [LLI-CAMS](https://github.com/CleilsonElias/LLI-CAMS) | `31cfe2f6c5720f2f24fe14c3fbfdee2b8cef190a` |
| [CALMS-SAVANT](https://github.com/CleilsonElias/CALMS-SAVANT) | `05c080fac6796ab57253167722f3cfe4c47d1613` |
| VAL | Referência local 0.1.0 e suplemento revisado de 864 execuções |

Não foram executados integração de bancos, treinamento de transformers ou
experimentos biológicos. RHAVENA e outros rascunhos não são classificados como
código publicado sem um repositório verificado nesta auditoria.

## λ designa controles diferentes

| Aspecto | LLI-CAMS 0.1.0 | VAL 0.2.0 | CALMS-SAVANT 0.1.1 |
|---|---|---|---|
| Função de λ | Inibição escalar de admissão | Inibição por domínio | Escala de inibição do aprendizado |
| Intervalo | [0,1] | [0,1] por domínio | λ > 0, inclusive maior que 1 |
| λ = 0 | Aceito | Aceito, sem inversão no portão | Rejeitado no gate e nos ranks |
| Seleção | `score >= threshold`, representantes | `h > threshold`, observações | Propostas de atualização governadas |
| Arquivo | Sombras, não representantes e demissões | Rejeitados no intervalo bruto | Assinaturas de recorrência |
| Objetivo | λ fixo neste núcleo | Ascensão de utilidade líquida externa | Descida de perda de treinamento |

Não existe conversão automática justificada entre esses parâmetros. Uma
integração precisa usar nomes distintos, como `lambda_admission` e
`lambda_learning`, declarar sua ligação e calibrá-la. Reutilizar valores não
demonstra equivalência dos mecanismos.

## Propagação das correções

| Alteração VAL | Efeito nos códigos anteriores | Decisão |
|---|---|---|
| Remover superioridade universal e métrica de consciência | Não muda sua execução; restringe inferências conjuntas | Documentar nos três projetos |
| Unificar score de admissão e arquivo | LLI já usa a mesma escala; CALMS não usa esse intervalo episódico | Alterar apenas o padrão VAL |
| Corrigir o sinal da adaptação de recompensa | CALMS já desce corretamente a perda; LLI não aprende λ | Preservar os sinais dos outros projetos |
| Calibrar baseline e urgência | LLI tem baseline global, sem amplificador dependente de λ | Não transplantar a EMA VAL |
| Limitar inversos e ranks | CALMS limitava o resultado depois de operações que podiam falhar | Corrigir overflow antes de potência/conversão |
| Separar plano de LoRA/REM de execução | LLI tem conectores opcionais; CALMS treina matrizes NumPy | Preservar as distinções de execução |
| Parear dados e contexto dos comparadores | Novos percentuais VAL não validam os outros sistemas | Manter estudos e resultados identificados |

### LLI-CAMS: nenhuma mudança de execução necessária

`src/llicams/sleep.py` usa o mesmo `selection_score` para ativos e sombras.
A configuração exige `shadow_threshold <= consolidation_min <= consolidation_max`.
Portanto, não há a mistura bruto/amplificado que criava a lacuna no VAL.

LLI pode arquivar um registro com score alto se ele não for representante de
cluster ou perder a vaga ativa. Impor `h <= threshold` a todas as suas sombras
removeria esses casos intencionais. Sua desigualdade `>=` também é preservada,
pois não equivale à desigualdade estrita do VAL.

`identity_depth_count_proxy`, em `engine.py`, calcula `A + log1p(S)`. Não é
a antiga soma ponderada por `1/λ` nem uma medida de consciência. O indicador
Eureka é condição de recuperação, não uma avaliação de novidade ou criatividade.
Releases, banco de dados, formatos e algoritmos LLI permanecem preservados.

### CALMS-SAVANT: correção numérica localizada

Antes do patch, `stratum_gate(1e-320)` gerava `OverflowError` apesar de o
resultado limitado dever ser 1. `allocate_ranks({'x': 1e-320}, max_rank=8,
budget=8)` falhava ao converter uma razão infinita para inteiro, embora o
resultado desejado fosse simplesmente rank 8.

Em `src/calms_savant/lora.py`, o gate agora retorna `optional_gate` diretamente
quando `0 < λ <= 1`, antes da potência inversa. Os ranks comparam a razão com
`max_rank` antes de `floor`/conversão inteira. Zero continua rejeitado. O
orçamento e o mínimo de um rank por domínio não mudam. O gate é testado também
com o menor float positivo representável.

O treino continua subtraindo o gradiente da perda. Inverter o sinal por analogia
com VAL faria o CALMS piorar seu objetivo. Transporte, arquivo de assinaturas,
projeções e suas limitações de segurança não mudam. A release 0.1.0 permanece
histórica; a correção é identificada como patch 0.1.1.

## Consequências científicas do estudo revisado

As 864 execuções usam 16 sementes, três cenários, dois níveis de erro e nove
políticas. Fatos, scores e alvos de consulta são iguais entre níveis de erro;
apenas o roteamento de ingestão muda. Todos têm teto de 128 episódios; bytes e
ocupação realizados são reportados. Os adaptativos recebem o mesmo sinal externo.

| Recall@1 sem erro de rota | Foco | Mudança de alvo | Difuso |
|---|---:|---:|---:|
| Limiar uniforme coerente | 39,91% | 64,91% | 38,80% |
| VAL seletivo fixo coerente | 73,83% | 55,99% | 39,65% |
| VAL adaptativo com contexto | 73,83% | 96,68% | 39,65% |
| Cotas fixas | 99,09% | 33,01% | 35,42% |
| Cotas adaptativas com mesmo contexto | 99,09% | 100,00% | 35,42% |

A especialização ajuda no teste focado, mas cotas simples obtêm resultado maior.
O VAL fixo perde desempenho relativo após a mudança. A ablação de amplificação
coerente não muda Recall@1 em nenhuma semente/condição. Esses resultados não
estabelecem superioridade universal, motivação aprendida, consciência ou
segurança de LLMs e não são evidência experimental dos outros projetos.

As duas proposições revisadas tratam da monotonicidade de admissão e de um traço
escalar não negativo com pesos fixos. Não provam monotonicidade da norma da
massa vetorial, densidade normalizada do grafo, utilidade ou capacidade cognitiva.
Essa restrição vale para qualquer leitura integrada dos trabalhos.

## Versões e reprodução

VAL 0.2.0 usa `raw_interval` em novas configurações e na demo. A opção `paper`
continua explícita. Snapshots 0.1.0 que salvaram essa opção mantêm a política
ao carregar; não existe migração automática de bancos entre projetos.

`studies/manuscript_revision/` conserva os oito arquivos do núcleo 0.1.0 sem
alterações e explicita a regra em cada política. O artigo permanece ligado a
esse núcleo congelado. O benchmark antigo da CLI permanece histórico. Nenhum
resultado antigo é reescrito como se tivesse usado a regra nova.

## Verificações executadas localmente

Python 3.12.14; skips correspondem a dependências opcionais ausentes.

| Conjunto | Aprovados | Skips | Falhas |
|---|---:|---:|---:|
| LLI-CAMS preservado | 42 | 3 | 0 |
| CALMS-SAVANT antes | 40 | 2 | 0 |
| CALMS-SAVANT corrigido | 44 | 2 | 0 |
| VAL 0.2.0 | 44 | 0 | 0 |
| Suplemento congelado | 6 | 0 | 0 |

Quatro novos testes CALMS cobrem subnormais, orçamento/teto, valores usuais e
inibição muito alta. Três novos testes VAL cobrem padrão, fronteiras exatas e
restauração de legado. A validação registra ainda uma comparação independente
da alocação antiga/nova em uma grade finita de configurações usuais.
CI remoto e componentes opcionais são verificações separadas.

Uma vantagem específica sobre cotas ainda exige dados reais, roteador calibrado
e orçamento comparável em bytes/tempo. Motivação aprendida exige um estimador
treinado e avaliação fora da calibração. Integração entre projetos exige testes
próprios de schema, interferência e objetivos.
