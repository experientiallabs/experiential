# Authored scenarios

Scenarios written or imported by a person, rather than mined from traces, join a completed build
through `exp.simulation.scenarios.select_authored_scenarios(project, scenarios)`. Each
`AuthoredScenario` carries an instruction, optional initial context, optional tools and an
optional reference answer. The build's task set becomes its mined tasks followed by the authored
ones; the trace dataset, retrieval indexes and world model are unchanged, and a later call
replaces the authored scenarios. Select them before authoring the judge, because a new task set
discards judge setup bound to the old one. `ModelEvaluationOptions(include_mined=False)` replays
only the authored scenarios; the default replays both. Judge calibration still samples real
traffic. Authored scenarios have no source traces, so their cost estimate assumes one answer
(sized from the reference answer, else 512 tokens) plus one tool call when tools are declared.
The judge sees a reference answer as its own section; candidates and the world model never do.
