## Core Development Rules

1. For any major code change involving model architecture, training strategy, or training data construction, record the reason, exact change, and timestamp in `docs/MODIFICATION_LOG.md` at the project root.

2. Before modifying core project logic, ask the user for confirmation and explain the intended change clearly.

3. Reading or analyzing files under the current project without changing code or file contents may proceed directly without extra confirmation. Provide clear terminal progress/output updates while doing so.

4. Do not delete existing content casually. Before deleting files, modules, functions, configs, comments that encode intent, or substantial code blocks, ask the user for confirmation first.

5. Prefer additive and reversible changes when exploring new ideas. Preserve existing baselines unless the user explicitly approves replacing or removing them.

6. After running new experiments or forming new experimental conclusions, record the experiment setup, key results, analysis, and next-step plan in a dedicated experiment-analysis document so the project retains memory across sessions. Prefer updating `docs/EXPERIMENT_RESULTS_AND_ANALYSIS.md` unless a more specific experiment document already exists.

7. DO NOT send optional commentary.

8. Before every code, configuration, training, data, or deployment change, read `docs/思考与隐患.md` and account for every unresolved item relevant to the change. When an item is resolved, update that document with the date, evidence, and affected scope; unresolved items remain mandatory considerations for subsequent changes.

9. Think comprehensively, but keep responses concise and clear. Communicate the core conclusions and necessary details without unnecessary complexity, repetition, or verbosity.

10. Before modifying code, verify that the change is necessary and appropriate. Review prior related changes and existing implementations first, reuse them when possible, and do not stack duplicate, overlapping, or redundant modifications.
