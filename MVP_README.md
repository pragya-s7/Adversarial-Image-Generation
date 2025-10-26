# Grounded Attention - MVP Quick Start

**Minimal Viable Product for Proof of Concept**

This is a streamlined implementation of the Grounded Attention mechanism for quick testing and iteration. The goal is to get baseline results fast and iterate from there.

## 🎯 MVP Goals

1. ✅ Implement core grounded attention mechanism
2. ✅ Integrate with LLaVA-1.5
3. ✅ Create minimal training pipeline
4. ✅ Set up basic evaluation
5. 🔄 Get initial results on hallucination benchmarks

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create conda environment
conda create -n grounded_attn python=3.10
conda activate grounded_attn

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt

# Install spaCy model for NLP
python -m spacy download en_core_web_sm
```

### 2. Test Core Modules

```bash
# Test grounded attention mechanism
python src/models/grounded_attention.py

# Test LLaVA integration
python src/models/llava_grounded.py

# Test loss functions
python src/training/losses.py

# Test dataset loading
python src/data/datasets.py
```

### 3. Download Data (Optional for Quick Testing)

For the MVP, you can start with a small subset of COCO:

```bash
# Create data directory
mkdir -p data/coco

# Download COCO 2014 validation set (smaller, good for testing)
wget http://images.cocodataset.org/zips/val2014.zip
unzip val2014.zip -d data/coco/

# Download annotations
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
unzip annotations_trainval2014.zip -d data/coco/
```

Or use a tiny subset for ultra-fast testing:

```bash
# Create a minimal test dataset
python -c "
from src.data.datasets import create_sample_annotation_file
create_sample_annotation_file('data/test_annotations.json', num_samples=10)
"
```

### 4. Run Simple Inference Test

```bash
# Test with base LLaVA model
python scripts/evaluate_simple.py \
    --model_name llava-hf/llava-1.5-7b-hf \
    --image_path /path/to/test/image.jpg \
    --prompt "Describe this image in detail."
```

### 5. Train on Small Dataset (Optional)

```bash
# Minimal training run for testing
python scripts/train_minimal.py \
    --data_root data/coco/val2014 \
    --annotation_file data/coco/annotations/captions_val2014.json \
    --output_dir outputs/mvp_test \
    --batch_size 2 \
    --num_epochs 1 \
    --max_samples 50 \
    --use_8bit
```

## 📁 Project Structure

```
grounded-attention/
├── src/
│   ├── models/
│   │   ├── grounded_attention.py      # Core grounding mechanism
│   │   └── llava_grounded.py          # LLaVA integration
│   ├── data/
│   │   └── datasets.py                # Dataset loaders
│   ├── training/
│   │   └── losses.py                  # Loss functions
│   └── utils/
├── scripts/
│   ├── train_minimal.py               # Minimal training script
│   └── evaluate_simple.py             # Simple evaluation
├── configs/
│   └── mvp_config.yaml                # MVP configuration
├── requirements.txt
├── setup.py
└── MVP_README.md                       # This file
```

## 🔧 Key Components

### 1. Grounded Attention Module

Located in `src/models/grounded_attention.py`

**Key Classes:**
- `GroundingHead`: Computes grounding scores for text tokens
- `GroundedCrossAttention`: Cross-attention with grounding mechanism

**Grounding Types:**
- `similarity`: Cosine similarity-based (recommended for MVP)
- `attention_weighted`: Weighted by attention distribution
- `learnable`: MLP-based learnable grounding

### 2. LLaVA Integration

Located in `src/models/llava_grounded.py`

**Key Functions:**
- `load_llava_with_grounding()`: Load LLaVA with grounding config
- `run_grounded_inference()`: Run inference with grounding

**Note:** Current MVP version is a proof-of-concept wrapper. Full integration requires modifying the decoder architecture (see TODO in code).

### 3. Loss Functions

Located in `src/training/losses.py`

**Available Losses:**
- `GroundingLoss`: Encourages high grounding scores
- `ContrastiveLoss`: Distinguishes grounded vs hallucinated
- `CombinedLoss`: Combines all losses

## 🎛️ Configuration

Edit `configs/mvp_config.yaml` to customize:

```yaml
model:
  grounded_layer_indices: [28, 29, 30, 31]  # Which layers to modify
  grounding_type: "similarity"              # Type of grounding
  grounding_strength: 1.0                   # Initial strength

training:
  batch_size: 4
  learning_rate: 2e-5
  lambda_grounding: 0.5                     # Grounding loss weight
```

## 📊 Expected MVP Results

**Goal:** Demonstrate that grounding mechanism can work in principle

**Success Criteria:**
- ✅ Code runs without errors
- ✅ Model can generate reasonable captions
- ✅ Grounding scores are computed correctly
- 🔄 Initial evidence of reduced hallucinations (even small improvement is good!)

**Not Expected in MVP:**
- State-of-the-art performance
- Full benchmark results
- Production-ready code
- Extensive ablations

## 🐛 Troubleshooting

### Out of Memory

```bash
# Use 8-bit quantization
--use_8bit

# Reduce batch size
--batch_size 1

# Reduce max samples
--max_samples 10
```

### Model Loading Issues

```bash
# Check HuggingFace cache
ls ~/.cache/huggingface/

# Clear cache if needed
rm -rf ~/.cache/huggingface/hub/*

# Re-download
python -c "from transformers import AutoModel; AutoModel.from_pretrained('llava-hf/llava-1.5-7b-hf')"
```

### Dataset Issues

```bash
# Use test data for quick iteration
python src/data/datasets.py  # Creates sample data in /tmp
```

## 📝 Next Steps (Post-MVP)

Once MVP is working:

1. **Full Integration**: Properly integrate grounding into decoder layers
2. **Data Generation**: Create hallucination dataset with positive/negative pairs
3. **Evaluation**: Run on POPE, CHAIR, MME benchmarks
4. **Optimization**: Tune hyperparameters, try different grounding types
5. **Ablation Studies**: Systematic comparison of design choices
6. **Scaling**: Test on larger models (13B, 70B)

## 🤝 Contributing

This is research code. Focus on:
- Getting results fast
- Documenting what works/doesn't work
- Iterating quickly

## 📚 References

- LLaVA: https://github.com/haotian-liu/LLaVA
- POPE Benchmark: https://github.com/AoiDragon/POPE
- CHAIR Metric: https://github.com/LisaAnne/Hallucination

## 💡 Tips for Fast Iteration

1. **Start Small**: Use 10-100 images for initial testing
2. **Use 8-bit**: Quantization makes things much faster
3. **Test Incrementally**: Test each component before training
4. **Monitor Losses**: Watch grounding scores during training
5. **Visualize**: Look at actual examples, not just metrics

## ⚡ Ultra-Fast Testing Workflow

```bash
# 1. Test core mechanism (30 seconds)
python src/models/grounded_attention.py

# 2. Test with dummy data (1 minute)
python src/data/datasets.py

# 3. Quick inference test (2 minutes)
python scripts/evaluate_simple.py --image_path test.jpg

# 4. Mini training run (5 minutes)
python scripts/train_minimal.py \
    --max_samples 10 \
    --batch_size 1 \
    --num_epochs 1
```

Good luck with your MVP! 🚀

---

**Remember:** The goal is to prove the concept works, not to achieve perfect results. Get something running, then iterate!
