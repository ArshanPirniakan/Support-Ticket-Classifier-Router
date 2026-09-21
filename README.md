
The confidence value above is illustrative formatting only; real values come from your own trained model.

---

## Why this project exists

Support desks need two different things from an automated triage system:

1. **Understanding what the customer is asking about.** This is a genuine NLP problem and is solved
   here with supervised multiclass text classification.
2. **Deciding who handles it and how urgent it is.** In most real organizations this is a policy
   decision, not a statistical one. It changes when the org chart or the SLA changes.

Many portfolio projects blur these two together and present rule outputs as if a model learned them.
This project keeps them explicitly separate:

- **Learned:** the intent/category label, trained on a real public dataset with real ground truth.
- **Not learned:** department and priority, produced by a small, readable, auditable rule layer.

The dataset used here does not contain department or priority ground truth, so no fake labels are
invented and no rule output is presented as a model prediction.

This is a real training and evaluation pipeline, not a wrapper around a hosted LLM API. No API keys
are required and no paid service is used.

---

## Architecture
