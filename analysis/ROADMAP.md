# ROADMAP  [WALK]

*phased. not all built yet. status marked per section.*

---

## phase 0 - grounding (done-ish)

- [x] `README.md` - what this is, who it's for
- [x] `notes/00_why_im_here.md` - orientation note
- [x] `quantization_lens.md` - the centerpiece research question, scoped
- [x] `atlas/build_leaderboard.py` + `leaderboard.csv` - survey of all 32 submissions
- [ ] `notes/01_techniques_seen.md` - running annotated list of techniques from the leaderboard
- [ ] `notes/02_papers.md` - reading list with one-line reactions per paper
- [ ] `notes/03_open_questions.md` - running questions, tagged by urgency

---

## phase A - technique atlas (reading, no compute)

*goal: understand what every submission actually did.*
*tier: REF - AI-assisted, i curate and verify. not what i defend in an interview.*

- [ ] `atlas/delta_chain.md` - narrative walkthrough: baseline (1.2244 BPB) to SOTA (0.9485),
      what each step bought, which authors handed off which ideas to whom
- [ ] `atlas/techniques/` - one file per technique family:
      - quantization: int8 row-wise, int6 GPTQ (Cholesky), ternary BitNet, asymmetric
      - optimizers: Muon, NeoMuon, AdamW per-tensor-type
      - architecture: depth/layer recurrence, parallel residuals (GPT-J), U-Net skips,
        XSA, SmearGate, weight tying
      - positional: RoPE, YaRN, learned QK-Gain
      - data: coprime-stride sharding, SentencePiece (1024/4096/8192), bigram hash
      - eval: sliding-window, score-first TTT, Muon-in-TTT, entropy-adaptive epochs
      - activations: LeakyReLU(0.5)^2, SwiGLU
      each page: plain-english explanation, paper link, code line in a submission, BPB delta
- [ ] `atlas/submissions/` - auto-stub all 32, hand-annotate top 10

---

## phase B - tiny reproductions (laptop GPU + some RunPod)

*goal: feel the numbers move. not SOTA, just intuition.*

- [ ] `notebooks/00_int8_math.ipynb` - derive scale/zero-point from scratch in numpy,
      verify against `quantize_state_dict_int8` in `train_gpt.py:~370`.
      this is the one piece i want to be able to do on a whiteboard. [OWN]
- [ ] `notebooks/01_baseline_walkthrough.ipynb` - annotated read of `train_gpt.py`.
      every block: what is it, why is it there.
- [ ] `notebooks/02_tiny_training_run.ipynb` - ~2-layer 128-dim config, 50MB FineWeb
      slice, target <5min on the 4070. don't touch train_gpt.py - just pass overrides.
- [ ] `notebooks/03_ablation_toggles.ipynb` - four switches:
      {tied_embeddings, rope_vs_learned_pos, int8_quant, muon_vs_adamw}.
      run each on/off pair on the tiny config. plot BPB. see which knobs move most.
- [ ] `notebooks/04_runpod_one_seed.ipynb` - one full-scale run on RunPod,
      single seed, ~$8-12 of the credit. pick the ternary submission because it's
      visually striking and the code is pedagogically dense.

---

## phase C - interpretability (laptop GPU, from saved checkpoint)

*goal: convert a trained checkpoint into a multi-angle case study.*

- [ ] `interp/01_attention_logitlens/` - hook residual stream, logit-lens across layers,
      visualize attention patterns on 5 hand-picked prompts
- [ ] `interp/02_probes/` - linear probes for:
      token position, POS (spaCy oracle), code-vs-prose, semantic topic
      ~50 lines each in scikit-learn on cached activations
- [ ] `interp/03_saes/` - tiny SAE on mid-layer residual stream.
      dictionary ~4x hidden. look at top-activating examples for ~10 features.
      honest caveat: don't know if they're "real" yet.
- [ ] `data_forensics/tokenizer_tour.ipynb` - SP8192 vocab dump, merge frequency,
      weird tokens
- [ ] `data_forensics/per_token_loss.ipynb` - which tokens dominate val_bpb?
      histogram of per-token loss. most ML beginners never look at this.
- [ ] `data_forensics/fineweb_sample_audit.md` - hand-read 50 random val documents.
      write what they ARE. boring, differentiating.

---

## phase D - the centerpiece experiment (compute-gated)

*see `quantization_lens.md` for full writeup.*

- [ ] `quantization_lens/pipeline.py` - train -> quantize at {fp32, int8, int6, ternary}
      -> cache activations -> run probes + SAE feature comparison
- [ ] `quantization_lens/results.md` - the feature x quantization-tier table.
      populated after running pipeline. [OWN]
- [ ] `quantization_lens/figures/` - BPB per tier, probe accuracy per tier,
      SAE activation frequency heatmap

---

## phase E - capstone

- [ ] `capstone.md` - 1500-2500 words. 10 things learned, 3 still confusing,
      3 follow-up questions. link out to every artifact. honest tone. [OWN]

---

## what i will NOT do

- try to land on the leaderboard. the artifact is the deliverable.
- pretend to own all 36 submissions. reference is enough.
- hide AI assistance. tools used: claude code, gemini cli, jules, codex.
  labeled per file.

---

## compute budget

| item | estimated cost | credit source |
|------|---------------|--------------|
| phase B tiny runs (laptop) | $0 | local 4070 |
| phase B runpod one seed | ~$10 | RunPod credit |
| phase C interp (laptop) | $0 | local 4070 |
| phase D quantization pipeline (laptop) | $0 | local 4070 |
| GCP experiments (if needed) | tbd | GCP credit |

*RunPod credit + GCP credit have expiry dates - spend those first on the
highest-value compute experiments.*
