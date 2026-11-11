from llicams import Config, Engine

with Engine("example.sqlite", Config()) as memory:
    memory.observe("O código fictício da amostra é AX-731.",
                   source="synthetic:example", metadata={"pinned": True})
    print(memory.sleep())
    result = memory.retrieve("Qual é o código da amostra?")
    for hit in result["hits"]:
        print(hit.data())
