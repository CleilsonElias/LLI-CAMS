# Publicação integral do código

## Conteúdo da release

O arquivo de distribuição contém todo o código desta implementação, incluindo
núcleo, aplicação, conectores opcionais, testes, documentação, exemplos sintéticos
e relatórios de validação. Não depende de arquivos das conversas de desenvolvimento.

O pacote não incorpora pesos de modelos nem bibliotecas externas. Eles devem ser
instalados separadamente, sob suas próprias licenças. A release de código pode ser
publicada integralmente; os dados futuros de pesquisa devem ser tratados como
outros artefatos, com sua própria política de divulgação.

## Verificação antes de uma release

```bash
python -m pip install -e .
python scripts/validate_release.py
python -m pip wheel . --no-deps --no-build-isolation --wheel-dir dist
```

`validation/validation.json` é um registro da execução, não uma certificação
independente. Preserve os resultados e a versão exata. Testes opcionais omitidos
devem continuar identificados na descrição da release.

## GitHub ou outro repositório

Crie o repositório na conta escolhida e envie este diretório. O nome de projeto no
pyproject é uma proposta; disponibilidade em PyPI não foi verificada nem reservada.
O workflow incluído descreve testes de núcleo em Python 3.11, 3.12 e 3.13; sua
presença não significa que esses jobs já tenham sido executados no GitHub.

Depois de criar o repositório, pode-se gerar a primeira tag:

```bash
git init
git add .
git commit -m "Release LLI-CAMS reference implementation 0.1.0"
git tag v0.1.0
```

Os comandos são instruções para execução pelo responsável. Não criam um destino
remoto ou publicam automaticamente. Não foi feito upload a uma conta externa.

## Depósito arquivável e citação

Associe a release a um depósito de software, por exemplo Zenodo, se desejar um
identificador persistente para o código. Atualize CITATION.cff com a URL real e
o DOI do software somente depois que forem emitidos. O DOI do artigo é mantido
como referência separada. Não reutilize um DOI do preprint como se identificasse
automaticamente esta versão de software.

O código foi implementado nesta release de novembro de 2026. A data de concepção
da arquitetura deve ser documentada pelos registros históricos próprios. Não
atribua aos arquivos atuais um timestamp de implementação de março ou abril.

## Descrição científica recomendada

Esta release disponibiliza uma implementação de referência de memória seletiva
para agentes, com seleção por entropia relacional, grafo ativo, sombras vetoriais,
recuperação condicionada e consolidação assíncrona. O núcleo foi verificado por
testes executáveis e benchmark sintético. Os conectores de transformer/LoRA são
opcionais e seu nível de validação está registrado nos artefatos da versão.
O pacote não demonstra consciência ou superioridade sobre sistemas existentes.

## Licença

A licença escolhida para esta implementação é MIT. Ela permite uso, modificação,
redistribuição e publicação com preservação do aviso de licença e copyright.
Os artigos anteriores e as dependências mantêm suas próprias licenças. O arquivo
NOTICE.md registra atribuição conceitual e assistência de IA na implementação.
