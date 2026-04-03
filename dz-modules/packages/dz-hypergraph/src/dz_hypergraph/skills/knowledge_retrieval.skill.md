---
name: knowledge_retrieval
description: Retrieve relevant known facts/theorems for a target statement.
---

# Knowledge Retrieval Skill

Return JSON only:

```json
{
  "facts": [
    {
      "statement": "...",
      "relation": "related|tool|obstruction|lemma",
      "source": "paper/book/library/known_result"
    }
  ]
}
```

Prefer precise, directly useful statements over generic background text.
