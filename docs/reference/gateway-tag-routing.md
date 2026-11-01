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
  policy is frozen; a later catalog change to a candidate makes the alias unavailable until the
  policy is frozen again.
- `classifier` is a `decision_model` document, exactly as a
  [classification guardrail](gateway-classification-guardrails.md) declares one: endpoint,
  model, categories with their questions, aggregate and threshold, window, timeout and cache.
  A category is flagged when its score reaches its threshold.
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
- When no rule matches, the default serves with `fallback_reason` `no_rule_matched`. When the
  decision model fails, times out, or the request has no text to classify, the default serves
  with `classifier_unavailable` or `no_classifiable_text`. Routing never fails a request; only
  an expired request deadline does.
- The first decision of an episode serves its later turns, as in learned routing, so a
  conversation keeps one model. Requests with no episode identity are routed one by one.
  Verdicts are cached by window, so the tool loop that follows one user message asks once.
- Selection runs on the gateway's callback loop, not in the learned selection lane, and is
  bounded by the classifier timeout and the request deadline.

Tag-rules policies serve through gateway aliases. Entry points that run only learned policies,
such as the optimizer's composition and replay, refuse them by name.
