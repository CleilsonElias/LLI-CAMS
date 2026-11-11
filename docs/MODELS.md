# Modelos e replay opcionais

## Encoder neural

Instale o extra semantic e use um modelo que você tenha escolhido e baixado,
respeitando sua licença. Fixe a revisão do modelo para reprodução.

```bash
python -m pip install -e ".[semantic]"
```

```python
from llicams import Config, Engine
from llicams.encoders import SentenceTransformerEncoder

encoder = SentenceTransformerEncoder("CAMINHO_OU_ID_DO_MODELO", revision="REVISAO_FIXADA")
config = Config(dimension=encoder.dimension)
with Engine("semantic.sqlite", config, encoder=encoder) as memory:
    memory.observe("Observação de exemplo", source="manual")
    memory.sleep()
```

As strings em maiúsculas são parâmetros que você deve substituir; não são modelos
ou revisões fictícias apresentados como existentes. Por padrão, o carregamento
usa apenas arquivos locais. A CLI oferece --allow-download para autorizar
explicitamente downloads e --encoder-model/--encoder-revision para a seleção.
Repita esses argumentos ao reabrir um banco com encoder neural.

## HNSW

```bash
python -m pip install -e ".[hnsw]"
```

Crie um banco com uma configuração JSON que contenha `"index": "hnsw"`. O índice
é reconstruído a partir dos vetores persistentes. Cada objeto publicado é
imutável; a aplicação não modifica o mesmo índice durante consultas concorrentes.
O parâmetro ef_search é 100; pedidos acima disso usam busca exata. Complexidade
logarítmica e recall perfeito não são garantias gerais do algoritmo.

## LoRA em um transformer real

```bash
python -m pip install -e ".[transformers]"
```

O módulo HFLoRAReplay implementa:

1. Seleção de textos ativos com `metadata.replay_approved: true`.
2. Carregamento de um modelo causal local, sem trust_remote_code.
3. Treinamento real de adaptador PEFT em linguagem causal nos textos aprovados.
4. Fusão por projeção contra a atualização acumulada anterior por camada.
5. Recompressão ao rank configurado, com registro de erro/sobreposição.
6. Gravação de adapter_model.safetensors, configuração e manifesto.
7. Retorno de um candidato para a publicação atômica do Engine.

Exemplo de execução com um modelo causal compatível e arquivos já disponíveis:

```bash
llicams --db neural.sqlite --backend transformers --model CAMINHO_DO_MODELO --replay-model CAMINHO_DO_MODELO --adapter-directory adapters --lora-rank 8 --lora-steps 20 serve
```

O exemplo usa CPU por padrão. --device altera o dispositivo, mas o software não
estima automaticamente se o modelo cabe na memória. O código carrega float32 e
não inclui receita de quantização. É uma referência inicial, não configuração
otimizada para um modelo grande ou uma GPU específica.

O modelo deve usar camadas compatíveis com target_modules q_proj e v_proj.
Na API Python é possível escolher outros alvos. A implementação é restrita a
LoRA padrão, alpha/r = 1, sem DoRA, embeddings adaptados ou vieses treináveis.

Para aprovar um registro, use JSONL:

```json
{"text":"Informação verificada pelo responsável.","source":"fonte do responsável","metadata":{"pinned":true,"replay_approved":true}}
```

O atributo não comprova verdade. H_R alto nunca aprova treino automaticamente.
O replay treina candidatos contra o modelo base; a fusão posterior aproxima a
combinação com o adaptador anterior. Isso não equivale a demonstrar treinamento
continual ótimo. Use conjuntos de retenção antigos e testes externos antes de
avaliar esquecimento ou generalização.

## O que foi validado nesta entrega

O ambiente desta entrega não continha PyTorch, Transformers, PEFT, modelos de
frases ou hnswlib. O núcleo e a matemática NumPy foram executados. O teste de
transformer pequeno está incluído e usa pesos aleatórios locais, sem download,
quando as dependências existem. Ele cobre treino, segunda fusão e carregamento,
mas foi omitido neste ambiente. Consulte validation/validation.json.

O conector Ollama foi testado com um servidor HTTP de fixture, não com um modelo
real respondendo. A interface de protocolo e a qualidade de um modelo são
validações diferentes.

## Fontes técnicas

- PEFT LoRA: https://huggingface.co/docs/peft/main/en/package_reference/lora
- PEFT state dict: https://huggingface.co/docs/peft/main/en/developer_guides/low_level_api
- Ollama chat: https://docs.ollama.com/api/chat
- HNSWlib: https://github.com/nmslib/hnswlib

Referências consultadas em 11 de novembro de 2026. Dependências futuras podem
exigir ajustes; registre versões e revisão do modelo em cada experimento.
