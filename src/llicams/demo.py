from dataclasses import replace
from pathlib import Path
from .config import Config
from .engine import Engine
from .llm import ask


FACTS = [
    ("Projeto fictício Aurora: o código da amostra de membrana é AX-731.", "amostra", "AX-731"),
    ("Projeto fictício Aurora: o protocolo de análise de proteína usa o código PR-208.", "protocolo", "PR-208"),
    ("O grupo fictício Vega trabalha com algoritmos para grafos de metabolismo celular.", None, None),
    ("O estudo fictício Órbita compara entropia de redes e fluxo de energia.", None, None),
    ("A equipe fictícia Delta associa memória de agentes a hipóteses verificáveis em experimentos.", None, None),
    ("A nota fictícia Ponte descreve uma relação entre controle de software e sinalização celular.", None, None),
    ("No projeto Aurora, a senha de identificação da réplica de laboratório é RP-884.", "replica", "RP-884"),
    ("Projeto fictício Íris: simulações de neurônios usam medidas de dinâmica e aprendizado.", None, None),
    ("O registro fictício Atlas contém dados sobre redes de proteína e computação por vetores.", None, None),
    ("O experimento fictício Nexo relaciona método de controle, energia e programas adaptativos.", None, None),
]


def run_demo(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    database = directory / "demo.sqlite"
    if database.exists():
        raise ValueError("Demo database already exists; choose a new directory to preserve the previous run")
    config = Config(consolidation_min=0.50, consolidation_max=0.82, shadow_threshold=0.20,
                    resonance_threshold=0.12)
    with Engine(database, config) as engine:
        admissions = []
        for i, (text, key, value) in enumerate(FACTS):
            metadata = {"pinned": i < 2}
            if key:
                metadata.update(fact_key=key, fact_value=value)
            memory = engine.observe(text, source="synthetic:demo", metadata=metadata, memory_id=f"demo-{i:02d}")
            admissions.append({"id": memory.id, "hr": memory.hr_normalized})
        report = engine.sleep()
        question = "Qual é o código da amostra de membrana do projeto Aurora?"
        return {"database": str(database), "observations": admissions, "sleep": report,
                "stats": engine.stats(), "query": ask(engine, question), "synthetic": True}
