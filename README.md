# LLI-CAMS Reference Implementation

**Versão 0.1.0 — implementação executável de referência para pesquisa.**

Este repositório implementa uma aplicação local de memória seletiva baseada em
LLI-CAMS: grafo ativo, arquivo de sombras, seleção por entropia relacional,
recuperação por ressonância, ciclo offline e publicação atômica de estado.
Inclui interface web local, CLI, API HTTP, testes, exemplo e benchmark sintético.

O código-fonte completo é distribuído sob a licença MIT. A arquitetura de
referência é de **Cleilson Elias de Sousa**. Veja `CITATION.cff`, `NOTICE.md` e
`docs/FORMALISM.md` para atribuição, limites e diferenças entre teoria e software.

## Executar em poucos passos

Requer Python 3.11 ou superior. O núcleo usa somente NumPy e a biblioteca padrão.
Não precisa de GPU, chave de API ou modelo baixado.

```bash
python -m venv .venv
```

Ative o ambiente no Linux/macOS:

```bash
source .venv/bin/activate
```

No Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Instale e execute:

```bash
python -m pip install -e .
llicams demo --directory demo_run
llicams --db demo_run/demo.sqlite serve
```

Abra **http://127.0.0.1:8765**. O exemplo contém somente registros fictícios.
Escolha outro diretório ao repetir `demo`; o comando preserva execuções existentes.

Sem instalar o pacote, com NumPy já disponível:

```bash
python run_llicams.py demo --directory demo_run
python run_llicams.py --db demo_run/demo.sqlite serve
```

## Usar seus próprios registros

```bash
llicams --db pesquisa.sqlite init
llicams --db pesquisa.sqlite observe "O código fictício da amostra é AX-731." --source "caderno de exemplo"
llicams --db pesquisa.sqlite sleep
llicams --db pesquisa.sqlite query "Qual é o código da amostra?"
llicams --db pesquisa.sqlite chat "Qual é o código da amostra?"
llicams --db pesquisa.sqlite stats
```

O backend padrão **retorna trechos recuperados**; ele não se apresenta como LLM.
Dependendo dos limiares, um registro pode ser descartado após consolidação. Use
`pinned` quando houver uma razão explícita para preservar um fato importante:

```bash
llicams --db pesquisa.sqlite observe "Código fictício AX-731." --metadata '{"pinned":true}'
```

No Windows, o uso de JSONL evita diferenças de escape entre shells:

```bash
llicams --db pesquisa.sqlite ingest examples/observations.jsonl
```

O JSONL aceita `text`, `source`, `confidence`, `metadata` e `id` opcional. Um ID já
existente é rejeitado. `confidence` é uma anotação fornecida pelo usuário, não uma
probabilidade de verdade inferida pelo sistema.

## O que está implementado

| Módulo | Implementação nesta versão |
|---|---|
| Entropia relacional | Equação original, forma equivalente O(K), normalização e verificações numéricas |
| λ-LI | Parâmetro configurável que controla o limiar; não é aprendido automaticamente |
| Grafo ativo I(t) | Nós persistentes e arestas de similaridade com grau limitado |
| Sombras S(t) | Direções normalizadas, peso ε, sequência temporal e modo de evidência explícito |
| Urgência Ω(q) | Sigmoide com referência histórica suavizada e prior configurável |
| Eureka | Nome operacional do disparo de recuperação; não implica insight útil demonstrado |
| Fase I SWS | Componentes por similaridade, seleção de representante e triagem |
| Fase II Replay | Hook real opcional para treinamento de LoRA com Transformers/PEFT |
| Fase III REM | Verificação estrutural limitada; geração e crítica por LLM opcionais |
| Fase IV Archival | Limites de capacidade e descarte de texto em `direction_only` |
| Alpha/Omega e ACS | Atendimento concorrente e preparação offline, com compare-and-swap SQLite |
| Exploração endógena | Perturbações controladas de consultas; candidatos não viram fatos automaticamente |
| HNSW | Backend opcional; busca exata é o padrão reproduzível |
| Persistência | SQLite, backup consistente, exportação e reabertura com verificação de configuração |
| Aplicação | Interface web responsiva, CLI e API local |

**A fase de replay aparece como `skipped` quando não está configurada.** A versão
não finge alterar pesos de um transformer no modo padrão.

## Dois modos de sombra

| Modo | Conteúdo retido | Consequência |
|---|---|---|
| `pointer` — padrão | Vetor e texto de evidência no registro persistente | Pode recuperar e citar o conteúdo; o custo do texto é contabilizado |
| `direction_only` | Vetor, campos escalares e referência de origem; texto e metadados do episódio são removidos do estado | Não reconstrói números, nomes ou fatos apagados por garantia; sinais latentes não são inventados como texto |

O modo padrão é uma **variante operacional com evidência**. Não corresponde ao
experimento estrito de eliminar a superfície episódica. A distinção é intencional
e aparece na documentação, nos resultados e na interface.

Exemplo de configuração alternativa:

```bash
llicams --db vetores.sqlite --config examples/direction_only.json init
llicams --db vetores.sqlite ingest examples/observations.jsonl
```

Depois da criação, a CLI recarrega a configuração armazenada no banco. Trocar
encoder, dimensão ou política exige outro banco ou migração explícita; misturas
silenciosas de espaços vetoriais são rejeitadas.

## Respostas com um LLM local

Com Ollama já instalado e um modelo já disponível no seu computador:

```bash
llicams --db pesquisa.sqlite --backend ollama --model NOME_DO_SEU_MODELO serve
```

`NOME_DO_SEU_MODELO` deve ser substituído por um nome real da sua instalação.
O código não baixa modelos ou cria serviços de terceiros automaticamente.
O adaptador chama `/api/chat` com `stream: false`. Não aplica adaptadores PEFT
ao processo do Ollama; para consumir o replay paramétrico, use Transformers.

Para encoder neural e replay, veja `docs/MODELS.md`. Os módulos opcionais têm
dependências e validação diferentes do núcleo. **Não houve execução de um LLM
real ou de treino de transformer no ambiente desta entrega.** O arquivo
`validation/validation.json` registra exatamente os testes executados e omitidos.

## Testar e reproduzir

```bash
python -m unittest discover -s tests -v
llicams benchmark --seeds 5 --observations 160 --queries 48 --output benchmark_results.json
python examples/linear_lora_demo.py
```

O benchmark mede recuperação de códigos factuais aleatórios com vetores
sintéticos e consultas de relevância tardia. Compara políticas de seleção,
recuperação sem urgência e ausência de texto nas sombras. Não mede criatividade,
consciência ou desempenho de linguagem natural. Os limites são iguais em número
de itens; bytes e uso de texto são reportados, não equiparados.

Resultados desta entrega ficam em `validation/`. Um resultado desfavorável a
H_R continua sendo um resultado válido; o código não exige que a proposta vença.

## Preservar e exportar estado

```bash
llicams --db pesquisa.sqlite backup --output backups/pesquisa.sqlite
llicams --db pesquisa.sqlite export --output memoria_exportada.json
llicams --db pesquisa.sqlite endogenous --count 3
llicams --db pesquisa.sqlite promote ID_DA_SOMBRA --evidence-note "Fonte conferida pelo responsável"
llicams --db pesquisa.sqlite forget ID_DA_MEMORIA
```

O backup usa a API de backup do SQLite; não copie apenas o arquivo `.sqlite`
enquanto houver escrita em WAL. Adaptadores de modelo são diretórios externos e
precisam acompanhar o backup se tiverem sido usados. `forget` remove a memória
externa, mas **não efetua unlearning de pesos já treinados**. Exportação JSON é
para inspeção/intercâmbio; restauração operacional usa o backup SQLite.

## Publicar o código integralmente

Este diretório pode ser publicado como repositório ou arquivo de release.
O pacote contém fontes, testes, exemplos, documentação e validação; não contém
suas conversas, documentos pessoais, bancos de pesquisa ou pesos de terceiros.
Veja o roteiro em `docs/PUBLISHING.md`. Nenhum upload a GitHub ou PyPI foi feito
por este projeto automaticamente.

## Limites científicos

O encoder padrão é lexical por hashing. Os protótipos são explícitos, não extraídos
de atenção de um transformer. O grafo expressa similaridade, não causalidade.
H_R alto não garante utilidade ou veracidade. O ACS local preserva registros
concorrentes por sequência; isso não demonstra conservação de entropia nem
disponibilidade ininterrupta em infraestrutura distribuída. Projeção ortogonal
de matrizes não garante ausência de interferência funcional. A contagem chamada
`identity_depth_count_proxy` é somente um índice estrutural.

Leia `docs/FORMALISM.md` e `docs/VALIDATION.md` antes de apresentar os resultados
como evidência científica.
