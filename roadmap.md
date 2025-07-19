<!-- # Markdown Cases

## Headers

```markdown
# H1
## H2
### H3
```

## Emphasis

*Italic*  
**Bold**  
***Bold Italic***

## Lists

- Unordered item
- Another item

1. Ordered item
2. Second item

## Links

[GitHub](https://github.com)

## Images

![Alt text](https://via.placeholder.com/100)

## Code

Inline: `code`  
Block:
```python
print("Hello, Markdown!")
```

## Blockquote

> This is a quote.

## Table

| Name   | Value |
|--------|-------|
| Foo    | 123   |
| Bar    | 456   |

## Task List

- [x] Task 1
- [ ] Task 2 -->

## Agents (Exclusive)
- agent1 $\longleftarrow$ factor investing
    - gpt4o
- agent2 $\longleftarrow$ mean regression
    - deepseek-reasoning
- ...
- agentn $\longleftarrow$ hedging
    - claude

## Support data (Shared)
- RAG $\longleftarrow$ market intelligence
    - news
    - media
    - important event
- tool-call $\longleftarrow$ financial tool
    - calculation
- LLM feature $\longleftarrow$ reasoning / CoT / VectorDB
- Base-model $\longleftarrow$ local deployment

## Multiagent collaboration
```mermaid
graph LR

A1(periodical price) --> B(multi-agent)
A2(mixed-scale price) --> B
B --> C1(choice 1)
B --> C2(choice 2)
B --> C3(choice 3)

C1 --> L(leader agent)
C2 --> L(leader agent)
C3 --> L(leader agent)
L --> D(final decision)
```