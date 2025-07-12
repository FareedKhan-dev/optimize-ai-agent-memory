# AI Agent Memory Optimization Cheatsheet

A quick reference guide to nine memory management techniques for building more efficient and intelligent AI agents.

---

### 1. Sequential Memory
-   **Core Idea:** Keep everything. Store the entire conversation history.
-   **How it Works:** Append every user and AI message to a growing list.
-   **Pros:** Perfect recall, simple to implement.
-   **Cons:** Extremely high token cost, quickly hits context limits, slow.
-   **Best For:** Short demos, simple bots, debugging.

---

### 2. Sliding Window Memory
-   **Core Idea:** Keep only the last 'N' conversation turns.
-   **How it Works:** Use a fixed-size queue (`deque`). As new messages are added, the oldest are automatically dropped.
-   **Pros:** Fixed and predictable token cost, fast, easy to manage.
-   **Cons:** Forgets all information outside the window, leading to a "context cliff."
-   **Best For:** General-purpose chatbots where recent context is most important.

---

### 3. Summarization Memory
-   **Core Idea:** Periodically create a running summary of the conversation.
-   **How it Works:** Use an LLM to combine a previous summary with recent messages from a buffer.
-   **Pros:** Balances context retention and token cost, maintains conversational flow.
-   **Cons:** Risk of losing details in the summarization process, extra LLM calls add cost and latency.
-   **Best For:** Long, creative conversations, story-telling, or brainstorming agents.

---

### 4. Retrieval-Based Memory (RAG)
-   **Core Idea:** Store conversation turns as searchable documents and retrieve only the most relevant ones.
-   **How it Works:** Embed each message and store it in a vector database (e.g., FAISS). On a new query, perform a similarity search to find the `top-k` most relevant past messages to use as context.
-   **Pros:** Highly efficient token usage, excellent for recalling specific facts, scalable.
-   **Cons:** Complex setup (embedding models, vector DB), retrieval quality is crucial.
-   **Best For:** Q&A bots, knowledge-based agents, modern production AI systems.

---

### 5. Memory-Augmented Transformers
-   **Core Idea:** Use a dual-memory system: a short-term window for recent chat and a long-term store for critical facts.
-   **How it Works:** An LLM acts as a "fact extractor" on each turn, creating concise "memory tokens" (e.g., "User is allergic to peanuts") that are stored permanently.
-   **Pros:** Excellent at retaining critical information, robust against context loss.
-   **Cons:** High compute cost due to extra LLM calls for fact extraction.
-   **Best For:** Reliable personal assistants, agents that must adhere to strict rules or user preferences.

---

### 6. Hierarchical Memory
-   **Core Idea:** Mimic human memory with short-term (working) and long-term (retrieval) layers.
-   **How it Works:** All messages go to a sliding window (working memory). If a message contains a keyword (e.g., "remember," "rule"), it's also "promoted" to a long-term retrieval memory.
-   **Pros:** Balances speed and deep recall, intuitive architecture.
-   **Cons:** Promotion logic can be either too simple (keywords) or costly (LLM-based).
-   **Best For:** Sophisticated agents that handle both casual chat and important facts.

---

### 7. Graph-Based Memory
-   **Core Idea:** Store information as a knowledge graph of entities and relationships.
-   **How it Works:** Use an LLM to extract `(Subject, Relation, Object)` triples from text. Add these as nodes and edges to a graph database (e.g., `networkx`).
-   **Pros:** Unparalleled for reasoning about connections, can answer complex, multi-hop questions.
-   **Cons:** Very high complexity to implement and maintain, extraction can be error-prone.
-   **Best For:** Expert systems, domain-specific knowledge bases, recommendation engines.

---

### 8. Compression & Consolidation Memory
-   **Core Idea:** Aggressively compress each turn into a dense, factual statement, stripping all "fluff."
-   **How it Works:** After each turn, use an LLM with a specific prompt to act as a "data compressor," rewriting the interaction into a single, concise fact.
-   **Pros:** Extremely token-efficient, great for factual recall on a tight budget.
-   **Cons:** Loses all conversational nuance, personality, and flow.
-   **Best For:** Task-oriented bots where only core facts matter (e.g., meeting notes, planning).

---

### 9. OS-Like Memory Management
-   **Core Idea:** Simulate a computer's memory with fast "RAM" (active context) and slower "Disk" (passive storage).
-   **How it Works:** Recent turns are kept in a small active buffer ("RAM"). When full, the oldest turn is "paged out" to passive storage. If a query needs old information, a "page fault" occurs, and the data is "paged in" to the context.
-   **Pros:** Conceptually powerful, allows for virtually limitless memory while keeping active context small.
-   **Cons:** Complex to manage the two tiers, "paging in" can be slow if not optimized.
-   **Best For:** Very long-running agents where conversations span days or weeks.

---

### Summary Table

| Technique                   | Complexity | Cost (Tokens/Compute) | Recall Quality                | Best For                                |
| --------------------------- | ---------- | --------------------- | ----------------------------- | --------------------------------------- |
| **Sequential**              | Very Low   | Very High             | Perfect (until limit)         | Demos, short chats                      |
| **Sliding Window**          | Low        | Low                   | Poor (for old info)           | General chatbots                        |
| **Summarization**           | Medium     | Medium                | Good (can lose detail)        | Creative/long conversations             |
| **Retrieval (RAG)**         | High       | Low                   | Excellent (for specifics)     | Q&A, expert agents                      |
| **Memory-Augmented**        | High       | High                  | Excellent (for critical facts)| Reliable personal assistants            |
| **Hierarchical**            | High       | Medium-High           | Very Good (layered)           | Sophisticated multi-task agents         |
| **Graph-Based**             | Very High  | High                  | Excellent (for relationships) | Knowledge-based systems                 |
| **Compression**             | Medium     | Low                   | Good (facts, no nuance)       | Task-oriented, factual bots             |
| **OS-Like**                 | Very High  | Low-Medium            | Very Good (if paged correctly)| Very long-running agents                |