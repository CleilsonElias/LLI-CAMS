# Validação e interpretação

Os resultados executados ficam em `validation/validation.json`,
`validation/unittest.txt`, `validation/benchmark.json` e
`validation/linear_lora.json`. O arquivo SUMMARY.md apresenta uma síntese gerada
dos mesmos dados. Execute `python scripts/validate_release.py` para recalcular.

## O que os testes verificam

- Equação original de H_R, limites, casos degenerados e normalização de vetores.
- Treino real de um adaptador linear de baixa ordem em NumPy, com queda da perda.
- Projeção ortogonal contra bases linearmente dependentes e resíduo de SVD.
- Seleção de estados, comportamento do gatilho e independência em relação a ε.
- Remoção de texto e metadados do estado de uma sombra direction_only.
- Preservação da evidência no modo pointer e ausência de reconstrução inventada.
- Reabertura, incompatibilidade de encoder/configuração, backup e remoção externa.
- Preservação de registros recebidos durante preparação lenta.
- Falha antes do commit e rejeição de candidato obsoleto entre dois Engines.
- Limites de capacidade e grau do grafo.
- API HTTP real em loopback, CLI, token, origem de escrita e agendamento opcional.
- Protocolo Ollama contra uma fixture HTTP local.

Os testes opcionais são omitidos quando faltam dependências ou modelo local.
O teste de transformer, quando habilitado, cria um modelo minúsculo aleatório e
um tokenizer local, executa replay e fusão duas vezes e tenta gerar com o artefato.
Ele é uma verificação de integração, não avaliação de linguagem ou conhecimento.

## Benchmark sintético

Cada stream contém fatos com códigos aleatórios. Vetores de observação e consulta
compartilham características de domínio e identidade; o mecanismo nunca recebe o
código correto como parte da consulta. As consultas são amostradas independentemente
do seletor H_R. O acerto exige recuperar texto que contenha o código esperado.

Há seis condições: paper, paper sem urgência, anchored, recência, aleatória e
direction_only. Todas têm os mesmos limites de itens ativos e sombras. Essa
igualdade **não é igualdade de bytes**: os custos de texto e vetor são reportados.
Os grupos não representam uma reprodução de Anthropic Dreams, MemGPT ou LightMem.

Uma condição com melhor resultado nesse teste pode ser pior em linguagem natural.
O número de sementes padrão é pequeno; desvios entre sementes são descritivos,
não prova de superioridade. Direções sem texto podem conservar afinidade e ainda
falhar na recuperação de códigos aleatórios. Esse resultado é informativo.

## O que não foi demonstrado

- Consciência, identidade psicológica, insight inevitável ou ausência de alucinação.
- Superioridade de H_R sobre sistemas de memória de fronteira.
- Prevenção de esquecimento em um transformer real ao longo de tarefas.
- Validade das hipóteses de seleção em dados humanos ou clínicos.
- Conservação de entropia por ACS ou disponibilidade distribuída ininterrupta.
- Desempenho em GPU ou escalabilidade para milhões de registros.

## Próximo experimento discriminante

Fixar um encoder semântico e modelo base, separar calibração e avaliação por
episódio, comparar recuperação de fatos, pistas tardias e pontes falsas sob o
mesmo orçamento total de bytes e computação. Só então adicionar replay LoRA e
medir retenção em tarefas antigas, conhecimento novo e alegações sem evidência.
Publicar resultados negativos e hiperparâmetros junto ao código.
