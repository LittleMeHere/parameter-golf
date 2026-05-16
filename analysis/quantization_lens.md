# quantization as a feature-stability lens  [OWN]

*alignment-adjacent now. alignment-direct if the methodology scales.*

---

## the question

as small LMs get compressed - fp32 to int8 to int6 to ternary - what
learned structure survives, and in what order does it die?

i keep framing this as an "alignment" question but i have to be careful
about what that means here. the param-golf base model is a raw FineWeb LM.
no RLHF. no instruction tuning. no refusal behavior. asking "does refusal
degrade under quantization" for THIS model would be a category error.

the right question for THIS model: which learned features (syntax structure,
token co-occurrence patterns, positional encoding fidelity, whatever the SAE
finds) are fragile under aggressive compression, which are robust, and is the
order predictable?

the alignment connection is downstream:
- same methodology, applied to an RLHF'd model
- now you can ask whether safety-critical features (refusal, harmlessness
  representations) degrade before or after capability-critical ones
- if safety degrades first, that's a flag for anyone shipping a quantized model

the param-golf run is a dry run. methodology on a clean, reproducible base.
real-model extension is fellowship-scale (or later).

---

## why i think this is worth doing

two empirical facts motivate it:

**1. quantization is already happening in production.**
int8 is standard. int4 is common. ternary is coming (BitNet, 2023). models
are being made smaller because smaller = cheaper to serve. the question is
not "should we quantize" but "what do we lose when we do."

**2. we don't actually know what dies first.**
there's decent engineering literature on how to quantize (GPTQ, AWQ, LLM.int8()).
there's almost no literature on WHICH representations inside the model survive
aggressive quantization. the interpretability community and the quantization
community are mostly not talking to each other yet. that gap is interesting.

---

## prior work i've read

**quantization methods (these are the how-to papers):**

- Frantar et al., "GPTQ: Accurate Post-Training Quantization for Generative
  Pre-trained Transformers." ICLR 2023.
  *how GPTQ works: layer-by-layer quantization with Hessian-based error
  compensation. this is the method used in most param-golf SOTA submissions.*

- Dettmers et al., "LLM.int8(): 8-bit Matrix Multiplication for Transformers
  at Scale." NeurIPS 2022.
  *key finding: outlier features in LLMs break naive int8. they decompose into
  mixed-precision (most weights int8, outlier dimensions fp16). this is directly
  relevant - "outlier features" are by definition high-magnitude, structurally
  important. what are they?*

- Ma et al., "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits."
  arXiv 2024.
  *BitNet b1.58 - ternary weights {-1, 0, 1}. gets surprisingly close to full
  precision. the question this paper doesn't answer: what specifically did the
  model give up?*

- Kim et al., "SqueezeLLM: Dense-and-Sparse Quantization." ICML 2023.
  *separates weights into a dense quantized part and a sparse high-precision
  part. the sparse part = outliers = the "important" weights. proxy for which
  parts of the network the quantization cares about.*

**interpretability (these are the what-is-inside papers):**

- Elhage et al., "Toy Models of Superposition." Anthropic 2022.
  *foundational. features can be packed into fewer dimensions than there are
  features - superposition. when you quantize, you're reducing numerical
  resolution. how does that interact with superposition?*

- Bricken et al., "Towards Monosemanticity: Decomposing Language Models With
  Dictionary Learning." Anthropic 2023.
  *sparse autoencoders (SAEs) can extract interpretable features from residual
  streams. this is the tool i want to use to measure "which features survive."*

**the gap:**
i have not found a paper that directly measures interpretable feature survival
across quantization tiers using SAEs or probes. that's either because (a) i
haven't found it yet, (b) it's been done privately inside a lab, or (c) it
genuinely hasn't been done at the scale / framing i'm interested in. i think
it's probably (a) and (c) mixed.

if you know of a direct reference here, open an issue - i'd genuinely like to know.

---

## hypothesis

stated so it could be wrong:

> features that encode high-frequency syntactic structure (short-range token
> dependencies, punctuation, whitespace) will survive aggressive quantization
> better than features that encode semantic or topical content.

why i think this: syntactic patterns appear many times per document and the
model should be able to encode them robustly with fewer bits. semantic features
(topic, entity, sentiment) might require finer precision because they're less
redundant in the data.

why i might be wrong: outlier features (the ones LLM.int8() identified as
structurally critical) are often positional / syntactic. if quantization
specifically preserves high-magnitude features, syntactic ones might actually
survive BECAUSE they're high-magnitude, not because they're redundant. the
causal direction matters.

secondary hypothesis: the ORDER in which features degrade is stable across
quantization methods (int8 vs int6 vs ternary), even if the RATE is not.

---

## proposed method

**step 0: pick the right checkpoint.**
train one seed of the param-golf baseline (or the laptop-scale tiny config)
to convergence. this is the fp32 reference.

**step 1: quantize to each tier.**
using code already in the param-golf codebase:
- int8: `quantize_state_dict_int8()` in `train_gpt.py` (line ~370)
- int6: GPTQ code from any mid-leaderboard submission
- ternary: BitNet-style code from `2026-03-24_74M_Ternary_UNet_...`

four checkpoints: fp32, int8, int6, ternary.

**step 2: cache activations.**
for each checkpoint, run the same 500-token prompt batch and cache the
residual stream at layer N/2 (midpoint). don't pick the output layer -
it's too task-specific. midpoint is where most of the "learned representation"
lives in smaller models.

**step 3: train linear probes.**
probes for:
- token position (easy baseline - should survive everything)
- part-of-speech tag (via spaCy oracle, ~30ms per doc)
- is-code vs is-prose (binary, robust to parse)
- semantic topic cluster (k-means on fp32 residuals, then probe across tiers)

train each probe on fp32 activations, then evaluate on quantized activations
WITHOUT retraining the probe. accuracy drop = feature degradation.

**step 4: train a tiny SAE.**
dictionary size ~4x hidden (so ~2048 for a 512-dim model).
train on fp32 residual stream. then measure activation frequency of each
learned feature across quantized models.

features that go quiet first = the fragile ones.
features that survive = the robust ones.

**step 5: write up the table.**

| feature | fp32 probe acc | int8 | int6 | ternary | dies at |
|---------|---------------|------|------|---------|---------|
| position | ... | ... | ... | ... | ? |
| POS | ... | ... | ... | ... | ? |
| code/prose | ... | ... | ... | ... | ? |
| topic | ... | ... | ... | ... | ? |
| SAE feat #N | ... | ... | ... | ... | ? |

---

## what i expect to find

- position probe survives everything (it's baked in by RoPE almost by definition)
- POS degrades somewhere between int6 and ternary
- topic degrades first (most abstract, least redundant in activations)
- SAE will find some features i don't have names for, and those are the
  interesting ones

---

## what would surprise me

- any feature surviving at ternary as well as int8 (would mean very
  redundant/high-magnitude encoding)
- syntactic features dying before semantic ones (would flip my hypothesis)
- the SAE failing to find stable features even at fp32 (would suggest the
  model is too small for features to be interpretable at all - also interesting)

---

## what i'd want to do next (fellowship-scale)

take this same methodology, apply it to a model that has been:
- RLHF'd
- constitutional AI'd
- safety fine-tuned with DPO or similar

now the probe set includes: refusal-relevant prompts, harmless vs harmful
classification, uncertainty expression.

if refusal representations degrade at int8 but capability representations
don't - that's a paper. that's the flag worth raising before someone ships
a quantized safety-trained model.

---

## honest limitations

- this is a toy model. FineWeb LM, not instruction-tuned, not RLHF'd.
  results may not generalize to larger models.
- my probe methodology is standard but not rigorous. linear probes are
  a blunt instrument. features that ARE present may not be linearly decodable.
- i haven't run this yet. everything above is a plan. the plan might break
  at step 2 if the laptop GPU can't hold four checkpoints in memory at once.
- i don't know what the SAE will find. nobody does until you run it.
  this is a feature, not a bug.

---

*status: scoped, not executed. compute sprint coming on RunPod credit.*
*estimated GPU time: ~2 hours for training + ~1 hour for probing, on RTX 4070.*
