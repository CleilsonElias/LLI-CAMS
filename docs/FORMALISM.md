# Relação entre o formalismo e a implementação

## Estado e espaço de representação

O estado publicado contém memórias ativas `I`, sombras `S`, arestas do grafo e
uma referência opcional a adaptador paramétrico. O buffer `B` contém observações
recebidas e ainda não processadas pelo ciclo offline. Elas podem ser consultadas
explicitamente como pendentes, sem fingir consolidação.

O encoder permanece fixo no banco. Seu fingerprint e a configuração são
verificados na reabertura. O encoder padrão é hashing lexical determinístico;
o encoder neural opcional usa Sentence Transformers. Ambos produzem vetores
unitários. O software não descobre automaticamente o valor causal de uma relação.

Os protótipos de K domínios são os embeddings de textos fornecidos na configuração.
As probabilidades são uma softmax dos cossenos multiplicados por uma temperatura
de afinidade. Esta é uma escolha operacional explícita. Não se trata de clusters
extraídos das matrizes de atenção de um transformer. O estágio SWS agrupa registros
pendentes por componentes conexos acima de um limiar de similaridade.

## Entropia relacional original

Para probabilidades p_i normalizadas:

\[
H_R(p)=-\sum_{i\ne j}p_i p_j\log_2(p_i p_j)
      =-2\sum_i p_i(1-p_i)\log_2p_i.
\]

`math.relational_entropy` usa a forma equivalente O(K), com 0 log 0 = 0.
Os testes comparam a implementação com a soma dupla original.

Os produtos fora da diagonal têm massa Z = 1 - sum(p_i²). Assim, H_R não é,
por si só, uma entropia conjunta normalizada sobre pares distintos. O limite
H_R <= 2 log2(K) continua válido. O núcleo usa:

\[
h_R=H_R/(2\log_2K).
\]

A divisão é uma convenção de escala, não uma correção que atribui utilidade ou
verdade à pontuação. Para K=4 uniforme, H_R=3 bits e h_R=0,75. Logo, limiares
acima de 0,75 podem bloquear toda consolidação não fixada nessa configuração.
Os valores padrão são pontos de partida de engenharia, não calibração empírica.

O seletor `paper` utiliza h_R. `anchored` utiliza h_R vezes a maior afinidade
positiva com um protótipo. O segundo é uma variante experimental, não uma
identidade algébrica com a fórmula original. Uma observação ortogonal a todos
os protótipos tem distribuição uniforme: `paper` pode selecioná-la, enquanto
`anchored` pode rejeitá-la. Esse caso está em teste para tornar a limitação visível.

## Inibição e triagem

\[
\theta(\lambda)=\theta_{min}+\lambda(\theta_{max}-\theta_{min}).
\]

λ-LI pertence a [0,1] e é fixado na configuração. Não há nesta versão um método
de meta-aprendizado que otimize λ ou uma política de Nash demonstrada.

1. Dentro de cada componente pendente, o representante elegível com maior
   prioridade pode ser consolidado se sua pontuação superar θ.
2. Outros registros elegíveis acima do limiar de sombra podem ir para S.
3. Os demais são descartados.
4. `pinned=true` é uma exceção operacional explícita para fatos que o usuário
   deseja preservar. Excesso de memórias fixadas produz erro de capacidade.
5. Limites de itens podem deslocar uma memória ativa para sombra e remover
   sombras. A prioridade de sombra combina pontuação, confiança anotada,
   ε e decaimento por sequência.

Essas regras tornam a aplicação limitada em recursos. Não afirmam que nenhuma
informação útil será perdida. O sistema não calcula o verdadeiro V_future.

## Referência histórica e urgência

\[
\gamma_n=1-e^{-n/T},\quad
\bar h_n=(1-\gamma_n)h_{prior}+\gamma_n\,\mathrm{mean}(h_{1:n}).
\]

\[
\Omega(q)=\sigma\{\alpha_\Omega[h_R(q)-\bar h_n]\}.
\]

O prior é configurável. O valor padrão não foi extraído de corpus de
pré-treinamento. A média histórica usa observações anteriores aceitas no buffer,
incluindo aquelas posteriormente descartadas. O contador continua válido após
evicções; reabrir um banco não reinicia silenciosamente a referência histórica.

## Recuperação e o nome Eureka

\[
\langle\widehat q,\widehat v_k\rangle\,\Omega(q)>\delta_{res}.
\]

Consulta e sombra são normalizadas. A desigualdade é estrita. ε não multiplica
essa condição, conforme a distinção do addendum entre peso subliminar e geometria
de busca. O arquivo fica fora do contexto de geração até ser recuperado.

`eureka=true` significa apenas que uma sombra passou pela condição. Não é um
teste de criatividade. Com Ω=0,5, uma sombra de cosseno alto pode disparar para
certos limiares sem um contexto excepcional. A seletividade precisa ser medida.

Consultas não promovem memórias nem reforçam pesos automaticamente. A promoção
externa exige ID e nota de evidência fornecida pelo responsável. Não existe
validador universal de fatos. Uma sombra sem texto não é promovida como lembrança
factual reconstruída, nem é enviada a um LLM como se o texto original existisse.

## Ciclo offline

| Fase | Execução |
|---|---|
| I SWS | Agrupamento de pendentes, representante, triagem e limites preliminares |
| II Replay | Treino opcional em memórias com replay_approved; candidato em diretório separado |
| III REM | Pares entre domínios e conflitos de fact_key/fact_value; geração/crítica opcionais |
| IV Archival | Estado final das sombras, remoção de texto em direction_only, grafo e limites |

O REM estrutural não gera descobertas: produz tarefas de avaliação de relações
e detecta somente conflitos anotados explicitamente. O REM por LLM faz duas
chamadas, geração e crítica; a crítica pode atenuar uma aresta. Suas propostas
ficam separadas e não entram automaticamente no acervo factual.

## Grafo e métricas

Arestas são cossenos positivos entre nós ativos, filtradas por limiar e por grau
máximo de cada nó. A construção densa custa O(A²D); a capacidade padrão A=256
mantém a rotina pequena, mas não resolve escalabilidade de grafos arbitrários.

O software não chama adjacência normalizada de Laplaciano. Não implementa uma
suposta equivalência entre o seletor original, diversidade de Rao e variação de
entropia espectral: são objetos diferentes que exigiriam experimentos separados.

O campo `identity_depth_count_proxy` implementa A + log(1 + S). Ele é um índice
de contagem, não evidência de identidade psicológica, consciência ou crescimento
irrestrito. Nenhuma variável C de consciência é calculada.

## LoRA e recompressão

O módulo opcional treina uma atualização real B A com PEFT. Em cada camada,
projeta a atualização candidata no complemento do span da atualização acumulada
anterior, usando o produto interno de Frobenius. A fusão é recomprimida por SVD
ao rank configurado. O manifesto registra o resíduo e a sobreposição restante.

Esta projeção é contra a atualização acumulada de uma camada, não contra todo o
histórico completo de subespaços anteriores. A recompressão pode desfazer
ortogonalidade. Mesmo uma projeção exata não demonstra ausência de interferência
nas saídas de um transformer. Preservação de tarefas antigas precisa de avaliação.

`fit_linear_lora` executa um treino numérico pequeno independente, comprovando a
execução dos gradientes de uma camada linear. Esse resultado não é apresentado
como experimento com LLM.
