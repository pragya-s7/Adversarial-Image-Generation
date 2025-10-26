# Grounded Attention - Quick Start Guide
## Get Running in 30 Minutes

This is your TL;DR version. For complete details, see the full project guide.

---

## The One-Sentence Pitch

We add a grounding score to transformer attention that penalizes tokens not visually supported, reducing hallucinations by 30%+ while maintaining model performance.

---

## What You Need

**Hardware**: 1-2 A100 GPUs (40GB+)  
**Time**: 12 weeks start to finish  
**Budget**: ~$2,000 for compute (if using cloud)

---

## Installation (5 minutes)

```bash
# Create environment
conda create -n grounded_attn python=3.10
conda activate grounded_attn

# Install core packages
pip install torch torchvision transformers accelerate peft
pip install datasets Pillow opencv-python pycocotools
pip install wandb spacy
python -m spacy download en_core_web_sm

# Clone starter code (when available)
git clone https://github.com/your-org/grounded-attention
cd grounded-attention
```

---

## The Core Idea (60 seconds)

**Problem**: VLMs hallucinate objects not in images  
**Why**: Standard attention doesn't enforce visual grounding  
**Solution**: Add grounding score $G(token, image)$ to attention weights  
**Implementation**: Modify cross-attention layers in language decoder

```python
# Standard attention
output = softmax(Q @ K.T / √d) @ V

# Grounded attention  
grounding_score = max_similarity(token_embedding, image_patches)
output = [softmax(Q @ K.T / √d) * sigmoid(grounding_score)] @ V
```

---

## Architecture Overview

```
┌─────────────────────────────────────────┐
│   LLaVA-1.5 Base Model                 │
│                                         │
│   Vision Encoder (CLIP ViT-L/14)      │
│            ↓                            │
│   Projection MLP                        │
│            ↓                            │
│   Language Decoder (Vicuna-7B)         │
│   ┌─────────────────────────────────┐  │
│   │ Layer 1: Standard Attention     │  │
│   │ Layer 2: Standard Attention     │  │
│   │ ...                              │  │
│   │ Layer 29: GROUNDED Attention ← NEW  │
│   │ Layer 30: GROUNDED Attention ← NEW  │
│   │ Layer 31: GROUNDED Attention ← NEW  │
│   │ Layer 32: GROUNDED Attention ← NEW  │
│   └─────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

**Key change**: Replace last 4 cross-attention layers with grounded versions.

---

## Implementation (3 Steps)

### Step 1: Grounding Function

```python
def compute_grounding_score(text_token, image_patches):
    """
    Returns: score ∈ [0, 1] indicating visual support
    """
    # Normalize embeddings
    token_embed = normalize(text_token)  # [D]
    patch_embeds = normalize(image_patches)  # [N, D]
    
    # Compute similarities
    similarities = token_embed @ patch_embeds.T  # [N]
    
    # Return max similarity
    return similarities.max()
```

### Step 2: Grounded Attention Layer

```python
class GroundedCrossAttention(nn.Module):
    def forward(self, text_features, image_features):
        # Standard attention
        attention_weights = softmax(Q @ K.T / √d)
        
        # Compute grounding scores
        grounding_scores = compute_grounding_score(text_features, image_features)
        
        # Modulate attention
        grounding_gate = sigmoid(grounding_scores)
        attention_weights *= grounding_gate
        
        # Apply to values
        output = attention_weights @ V
        return output
```

### Step 3: Training Objective

```python
# Total loss
loss = lm_loss + λ_ground * grounding_loss + λ_contrast * contrastive_loss

# Where:
# - lm_loss: Standard language modeling
# - grounding_loss: Penalize low grounding scores for nouns
# - contrastive_loss: Separate grounded from hallucinated examples
```

---

## Data You Need

**Positive Examples** (grounded):
- COCO Captions: 118K images → `/data/coco/`
- Visual Genome: 108K images → `/data/vg/`

**Negative Examples** (hallucinated):
- Generate using GPT-4: 50K examples
- Object swapping: 30K examples
- Attribute changes: 20K examples

**Quick generation**:
```python
# Swap object
original = "A dog sitting on grass"
hallucinated = "A cat sitting on grass"  # if no cat in image

# Change attribute  
original = "A red car parked"
hallucinated = "A blue car parked"  # if car is red
```

---

## Training Recipe

```python
# Hyperparameters (starting point)
batch_size = 16
learning_rate = 1e-5
epochs = 10
λ_grounding = 0.5
λ_contrastive = 0.1

# Use LoRA for efficiency
lora_r = 16
lora_alpha = 32
target_modules = ["q_proj", "v_proj"]

# Training command
python scripts/train.py \
    --model llava-1.5-7b \
    --data_dir /data/coco \
    --output_dir /outputs/grounded_llava \
    --batch_size 16 \
    --learning_rate 1e-5 \
    --epochs 10 \
    --use_lora \
    --lambda_grounding 0.5 \
    --lambda_contrastive 0.1
```

**Expected training time**: 8-10 hours per epoch on 1x A100  
**Total time**: ~100 GPU hours for full training

---

## Evaluation (Run These)

### POPE (Object Hallucination)
```bash
python scripts/evaluate.py \
    --benchmark pope \
    --model /outputs/grounded_llava/best_checkpoint \
    --data /data/pope
```
**Target**: >90% accuracy (baseline: ~85%)

### CHAIR (Caption Hallucination)
```bash
python scripts/evaluate.py \
    --benchmark chair \
    --model /outputs/grounded_llava/best_checkpoint \
    --data /data/coco
```
**Target**: <20% CHAIR-I, <5% CHAIR-S (baseline: ~30%, ~10%)

### MME (General Capability)
```bash
python scripts/evaluate.py \
    --benchmark mme \
    --model /outputs/grounded_llava/best_checkpoint \
    --data /data/mme
```
**Target**: Maintain or improve on baseline

---

## Must-Have Ablations

1. **Where to ground?**
   - Last 1 layer vs. last 4 vs. all 32 layers
   
2. **What grounding function?**
   - Max similarity vs. attention-weighted vs. learnable MLP
   
3. **How much penalty?**
   - λ ∈ {0.1, 0.5, 1.0, 2.0}
   
4. **Does contrastive help?**
   - With vs. without negative examples
   
5. **Model size?**
   - 7B vs. 13B parameters

---

## Timeline (12 Weeks)

| Week | Tasks | Deliverable |
|------|-------|-------------|
| 1 | Setup, baseline | Working environment, baseline scores |
| 2 | Implement grounding | Core mechanism working |
| 3-4 | Generate negatives, train | Trained model v1 |
| 5-6 | Evaluate, ablate | Full results table |
| 7-8 | Theory, analysis | Failure cases, theory section |
| 9-10 | Write paper | Draft paper |
| 11 | Polish | Camera-ready |
| 12 | Submit | CVPR submission |

---

## Success Metrics

**Must achieve**:
- ✅ 20%+ reduction in CHAIR
- ✅ 10%+ improvement in POPE
- ✅ No drop on MME
- ✅ Working code + trained models
- ✅ 8-page paper with ablations

**Nice to have**:
- ⭐ 30%+ reduction in CHAIR
- ⭐ 15%+ improvement in POPE
- ⭐ SOTA on hallucination benchmarks
- ⭐ Attention visualizations
- ⭐ Theory proof/lemma

---

## Common Issues & Fixes

**Issue**: Training instability  
**Fix**: Reduce learning rate to 5e-6, freeze vision encoder, use gradient clipping

**Issue**: Grounding doesn't help  
**Fix**: Try learnable grounding head, increase λ_grounding, check data quality

**Issue**: Fluency drops  
**Fix**: Reduce λ_grounding, only apply to nouns, add fluency metric

**Issue**: Out of memory  
**Fix**: Use LoRA, reduce batch size, use 8-bit quantization

**Issue**: Slow inference  
**Fix**: Only ground last 2 layers, cache grounding scores, optimize similarity computation

---

## File Structure (Minimal)

```
grounded-attention/
├── configs/
│   └── train.yaml          # Hyperparameters
├── src/
│   ├── models/
│   │   ├── grounded_attention.py    # Core mechanism
│   │   └── llava_grounded.py        # Integration
│   ├── data/
│   │   └── datasets.py              # Data loading
│   └── training/
│       └── losses.py                # Loss functions
├── scripts/
│   ├── train.py                     # Training script
│   └── evaluate.py                  # Evaluation script
└── README.md
```

---

## Quick Commands

```bash
# Train
python scripts/train.py --config configs/train.yaml

# Evaluate POPE
python scripts/evaluate.py --benchmark pope --checkpoint outputs/best

# Evaluate CHAIR  
python scripts/evaluate.py --benchmark chair --checkpoint outputs/best

# Generate negatives
python scripts/generate_negatives.py --input data/coco --output data/negatives

# Visualize attention
python scripts/visualize.py --image test.jpg --model outputs/best
```

---

## Paper Writing Priorities

**Week 9**: Draft structure
- Abstract (1 hour)
- Intro + motivation (4 hours)
- Method + architecture diagram (8 hours)

**Week 10**: Experiments + figures
- Results tables (4 hours)
- Ablation tables (4 hours)
- Qualitative figures (8 hours)

**Week 11**: Theory + polish
- Information theory section (6 hours)
- Related work (4 hours)
- Conclusion + limitations (2 hours)
- Revisions + formatting (4 hours)

---

## Key Figures to Create

1. **Teaser**: Side-by-side hallucination examples
2. **Architecture**: Grounded attention diagram
3. **Results**: Bar chart (POPE, CHAIR, MME)
4. **Ablations**: Line plots for λ sweep
5. **Attention maps**: Visualization of grounding scores
6. **Qualitative**: Grid of example outputs

---

## Critical Reminders

⚠️ **Document everything** - Future you needs notes  
⚠️ **Track all experiments** - Use W&B religiously  
⚠️ **Version control** - Commit code daily  
⚠️ **Backup checkpoints** - Cloud storage for models  
⚠️ **Test incrementally** - Don't wait for full training  

---

## Emergency Contacts

- **Stuck on code?** → Check full project guide Section 5
- **Unsure about experiments?** → See Section 6
- **Theory questions?** → Read Section 7
- **Paper structure?** → Review Section 8
- **Need motivation?** → Remember: This could be a Best Paper! 🏆

---

## First Action Items (Do These Now)

1. [ ] Setup environment (30 min)
2. [ ] Download LLaVA-1.5-7B (10 min)
3. [ ] Run inference test (5 min)
4. [ ] Read core grounding code (20 min)
5. [ ] Generate 10 negative examples (15 min)
6. [ ] Train on debug dataset (10 samples, 10 min)
7. [ ] Evaluate on 10 POPE samples (5 min)

**Total**: ~90 minutes to working prototype!

---

## Resources

- **Full Guide**: `grounded_attention_project_guide.md` (35,000 words)
- **LLaVA Repo**: github.com/haotian-liu/LLaVA
- **POPE Benchmark**: github.com/AoiDragon/POPE
- **CHAIR Metric**: github.com/LisaAnne/Hallucination
- **Discord**: [Your team channel]

---

## Final Motivation

You're not just implementing a paper—you're solving a fundamental problem in AI safety. Every percentage point reduction in hallucinations could prevent misinformation, improve accessibility tools, and make autonomous systems safer.

**This matters. Do it well. Win that award.** 🎯

---

**Now go execute!** Start with the first action items above, then dive into the full project guide for details. You've got this! 💪
