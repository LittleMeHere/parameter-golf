# why I'm here  [OWN]

mood: deciding where to focus.

---

option (a) actual ML eng. making AI more capable. physics, music,
  harder reasoning. loved this for months. still love it.

option (b) safety + alignment.

leaning b. why →

- there are ALREADY tons of brilliant engineers on (a). that side will be fine.
- alignment is the bottleneck on whether ANY of (a) ships outside the lab.
  capability without alignment doesn't ship. or it ships and gets pulled.
- was complaining about my own guardrails the other day and the question
  hit: *why aren't YOU doing something about this?*
- so. doing something about it.

position: AI has got to stop being intelligent without judgement.

---

why interp specifically (probably) →

manufacturing background. five-axis CNC, that world.
when something's broken, I open the manual. you can do that with a CNC.
you can't do that with ChatGPT. yet.

mech interp = closest thing to writing the manual.

---

starting with: quantization as a feature-stability lens.
*alignment-adjacent now, alignment-direct if this methodology scales.*

as small LMs get crushed (int8 → int6 → ternary), what learned structure
dies first? is the order predictable from fp32 metrics? stable across
compression methods, or method-dependent?

dry run on the param-golf base model - no RLHF there, so no refusal behavior
to measure directly. methodology is the point. once it works on a base model
it ports to real instruction-tuned / safety-trained models at larger scale.

writeup coming in `quantization_lens.md`.

---

caveat: not locked in. the field is huge:
  - red teaming
  - evals
  - interp ← leading
  - model organisms
  - governance
  - economics of these systems

scope before commit. ask future-me in a month.
