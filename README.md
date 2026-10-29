# MSA Systems Lab

**Sparse Attention · GPU Execution · Training Systems**

An independent engineering lab exploring how block-sparse attention moves from mathematical semantics to correct and efficient execution.

The project separates three questions:

1. **Correctness:** Do the masked-dense oracle and gathered sparse reference agree on the same causal support?
2. **Training:** What does learned routing change in downstream task behavior?
3. **Performance:** When does fewer QK/PV operations actually reduce GPU execution time?

## Engineering investigations

- **Fewer FLOPs, slower execution.** A transparent gathered implementation can reduce attention arithmetic yet incur additional host/device synchronization, irregular indexing, and kernel launches.
- **Routing quality vs. task quality.** Router loss, retained attention mass, token NLL, and top-1 correctness answer different questions.

Evidence-backed case studies and carefully selected source/tests will be published incrementally. Results from recorded internal experiments are not yet independently reproducible from this public repository.

## Status

| Area | Status |
| --- | --- |
| Causal support, reference forward, manual backward | Internally verified; public code pending |
| Learned routing and small-model training diagnosis | Internally verified; public evidence pending |
| GPU kernel parity and measured acceleration | Future work |

This is a technical showcase rather than a mirror of private learning materials. Optimization work will retain transparent references as correctness oracles.

> Independent reproduction and systems investigation. Not an official MiniMax implementation, a production library, or a claim of a new attention algorithm.
