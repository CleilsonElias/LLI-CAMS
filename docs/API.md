# API HTTP local

Base padrão: http://127.0.0.1:8765. Escritas exigem Content-Type application/json.
Se LLICAMS_API_TOKEN estiver configurado, envie Authorization: Bearer TOKEN.
O corpo máximo é 1 MiB. O serviço não oferece TLS nem contas de usuário.

| Método | Rota | Uso |
|---|---|---|
| GET | / | Interface local |
| GET | /api/health | Estado do serviço e tipo de backend |
| GET | /api/state | Estatísticas, registros sem vetores e propostas |
| GET | /api/export | Exportação completa para inspeção, incluindo vetores |
| POST | /api/observe | Receber text, source, confidence e metadata |
| POST | /api/query | Recuperar memória por query e k |
| POST | /api/chat | Recuperar e responder usando o backend configurado |
| POST | /api/sleep | Preparar consolidação; retorna job_id |
| GET | /api/jobs/ID | Consultar running, completed ou failed |
| POST | /api/promote | Promover sombra com id e evidence_note |
| POST | /api/endogenous | Gerar candidatos exploratórios com count |

Exemplo de registro:

```json
{
  "text": "Código fictício da amostra AX-731.",
  "source": "synthetic:manual",
  "confidence": 1.0,
  "metadata": {"pinned": true}
}
```

Exemplo de consulta:

```json
{"query": "Qual é o código da amostra?", "k": 5}
```

A resposta de consulta contém versão, H_R, referência histórica, Ω e hits.
Cada hit identifica estado, origem, texto recuperável, similaridade e disparo
de sombra. `eureka: true` é um evento do mecanismo, não certificação da resposta.

Escritas malformadas retornam 400; tipo de conteúdo incorreto, 415; conflito de
versão ou capacidade, 409; token ausente/incorreto, 401. Um ciclo de sono que falha
é mostrado como failed sem substituir a versão publicada.

A API não aceita um caminho para ler arquivos do computador, nem URLs de backend
arbitrárias no corpo da requisição. Ingestão de arquivos é feita pela CLI.
