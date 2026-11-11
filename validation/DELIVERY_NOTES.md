# Limites da validação da entrega

As verificações executadas estão em validation.json e unittest.txt. O núcleo,
CLI, servidor HTTP, scheduler e protocolo de integração Ollama foram executados.
Os exemplos e o benchmark usam somente dados sintéticos.

O ambiente não tinha os backends opcionais Torch/Transformers/PEFT, Sentence
Transformers ou HNSWlib. Seus testes foram explicitamente omitidos. O replay real
de um transformer está implementado, mas não foi executado nesta entrega.

A verificação visual/interativa em navegador não foi concluída: o processo
Chromium local não pôde iniciar e o navegador remoto não alcançou o servidor
de loopback do ambiente. As rotas da aplicação foram testadas por HTTP real.
Não se apresenta essa verificação de API como inspeção visual do layout.

Os resultados sintéticos não favorecem necessariamente H_R. Na execução desta
release, o benchmark apresenta as métricas de todas as condições, incluindo
controles mais fortes que o seletor proposto. Não houve ajuste posterior dos
hiperparâmetros para produzir vitória da arquitetura.
