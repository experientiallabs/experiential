# Tag routing

A tag-rules router policy picks the model for each request from a decision model's answers.
The decision model scores the request before it is dispatched, the first matching rule names a
candidate, and a default covers everything else. It is a project activation like a learned
router policy, so an alias activates it the same way.

Use it when the routing rule is something you can state, such as "cyber work goes to model A,
other coding work to model B". Use a learned policy (`exp optimize router`) when the choice
should come from measured quality and cost.

## The rules file

```json
{
  "candidates": ["coder", "secure", "generalist"],
  "default": "generalist",
  "classifier": {
    "adapter_id": "task-tags",
    "kind": "decision_model",
    "url": "http://127.0.0.1:30000/systemone",
    "model": "Cloudflare/clef",
    "categories": {
      "coding": {"threshold": 0.7, "questions": {"coding_work": {"instructions": "Is the latest user message asking for coding work?"}}},
      "cyber": {"threshold": 0.8, "questions": {"security_work": {"instructions": "Is the latest user message about computer security?"}}}
    }
  },
  "rules": [
    {"when_all": ["cyber"], "route": "secure"},
    {"when_all": ["coding"], "when_none": ["cyber"], "route": "coder"}
  ]
}
```

- `candidates` are aliases in `ROOT/models.toml`. Each one's model identity is pinned when the
  policy is frozen. An active alias keeps serving the catalog snapshot it was created with; after
  a catalog change to a candidate, freeze the policy again before the next `alias create`, which
  refuses a candidate whose identity changed.
- `classifier` is a `decision_model` document, exactly as a
  [classification guardrail](gateway-classification-guardrails.md) declares one: endpoint,
  model, categories with their questions, aggregate and threshold, window, timeout and cache.
  A category is flagged when its score reaches its threshold. The classifier names no
  credential (`bearer_env` and URL query strings are refused): a frozen policy carries none,
  so point it at a decision-model server that needs no token, such as one on the same host.
- A rule matches when every `when_all` category is flagged and no `when_none` category is.
  Rules apply in order and the first match wins. Every rule must route to a candidate and name
  only the classifier's categories.

## Freeze and activate

```bash
exp config gateway router tags rules.json --project PROJECT
exp config gateway alias create ALIAS --project PROJECT --policy POLICY_ID
```

`router tags` stores the policy in an existing project and prints its ID. The ID covers the
pinned candidates, the classifier, the rules and the default, so freezing the same file against
the same catalog again changes nothing. `alias create` checks the candidates against the catalog
and activates the alias.

## At request time

- The decision model reads its window of the request once, before dispatch. Tag routing is
  serial: the model has to be chosen before it is called.
- The selected candidate's exact model and certified pool serve the request, with
  `route_reason` `tag_router`.
- A selected candidate whose declared capabilities rule out the request (tools, or an output
  token limit below the request) yields to the default, then the first eligible candidate, with
  `fallback_reason` `capability_eligibility`, as in learned routing. The episode then stays on
  the candidate that served it.
- When no rule matches, the default serves with `fallback_reason` `no_rule_matched`. When the
  decision model fails or times out, when 32 classifications are already in flight across the
  gateway, or when the request has no text to classify, the default serves with
  `classifier_unavailable`, `classifier_busy` or `no_classifiable_text`. Routing never fails a
  request; only an expired request deadline does.
- An episode's decision serves its later turns, as in learned routing, so a conversation
  keeps one model while it stays active; an idle episode expires after a day, or
  sooner when many other episodes arrive. A default chosen because the decision model had no
  answer is not kept for the episode, so the next turn asks again. Requests with no episode
  identity are routed one by one. Verdicts are cached by window, so the tool loop that follows
  one user message asks once.
- A request whose deadline runs out while the decision model is answering fails, as in learned
  routing.
- Selection runs on the gateway's callback loop, not in the learned selection lane, and is
  bounded by the classifier timeout and the request deadline.

Tag-rules policies are stored as their own artifact type, so the optimizer's commands never see
them, and they serve through gateway aliases.
