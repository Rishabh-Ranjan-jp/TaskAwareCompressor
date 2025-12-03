"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                          CONFIG.PY                                          ║
║            Dashboard Configuration & Smart Model Loading                    ║
║                                                                              ║
║  Controls which models are loaded (binary or multiclass)                    ║
║  Manages all metrics, paths, and decoder information                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
from pathlib import Path
from typing import Dict, Literal

# ═══════════════════════════════════════════════════════════════════════════════
# 📁 PROJECT PATHS
# ═══════════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).parent
CUSTOM_DATASET_PATH = PROJECT_ROOT / "custom_dataset"
MODELS_PATH = PROJECT_ROOT / "Decoders"
MULTICLASS_MODELS_PATH = PROJECT_ROOT / "MulticlassExpansion"
CLASSIFIERS_PATH = PROJECT_ROOT / "classifiers"

# ═══════════════════════════════════════════════════════════════════════════════
# 🎯 CRITICAL: LOAD MODE SELECTOR
# ═══════════════════════════════════════════════════════════════════════════════

LOAD_MODE: Literal["binary", "multiclass"] = "binary"

"""
┌─ LOAD_MODE Controls Which Models Load ─────────────────────────────┐
│                                                                     │
│  "binary"      → For custom_dataset (Cats vs Dogs)                 │
│  ✅ Loads: 4 binary decoders + binary classifier (1.5 GB)          │
│  ✅ Shows: Cat/Dog classes only                                     │
│  ✅ Use at: Project expo with custom images                         │
│                                                                     │
│  "multiclass"  → For full CIFAR-10                                 │
│  ✅ Loads: 4 multiclass decoders + multiclass classifier (3.5 GB)  │
│  ✅ Shows: All 10 CIFAR-10 classes                                 │
│  ✅ Use for: Comprehensive results demo                             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
"""

# ═══════════════════════════════════════════════════════════════════════════════
# 📊 CLASS DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

BINARY_CLASSES: Dict[int, str] = {
    3: "Cat",
    5: "Dog"
}

CIFAR10_CLASSES: Dict[int, str] = {
    0: "Airplane", 1: "Automobile", 2: "Bird", 3: "Cat", 4: "Deer",
    5: "Dog", 6: "Frog", 7: "Horse", 8: "Ship", 9: "Truck"
}

# ═══════════════════════════════════════════════════════════════════════════════
# 🔧 NORMALIZATION CONSTANTS (from Images.ipynb)
# ═══════════════════════════════════════════════════════════════════════════════

CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.247, 0.243, 0.261]

# ═══════════════════════════════════════════════════════════════════════════════
# 🎨 BINARY DECODER CONFIGURATIONS
# ═══════════════════════════════════════════════════════════════════════════════

BINARY_DECODERS: Dict[str, Dict] = {
    "Simple Tanh": {
        "path": MODELS_PATH / "Task-Aware" / "SimpleTanh0.01.pth",
        "architecture": "Simple",
        "activation": "tanh",
        "alpha": 0.01,
        "accuracy": 0.7920,
        "ce_loss": 2.2177,
        "compaction_pct": 22.43,
        "description": "Simple decoder with Tanh - maximum compaction"
    },
    "Simple Sigmoid": {
        "path": MODELS_PATH / "Task-Aware" / "SimpleSigmoid0.20.pth",
        "architecture": "Simple",
        "activation": "sigmoid",
        "alpha": 0.20,
        "accuracy": 0.7870,
        "ce_loss": 2.2729,
        "compaction_pct": 23.24,
        "description": "Simple decoder with Sigmoid - highest compaction"
    },
    "MR Tanh": {
        "path": MODELS_PATH / "Task-Aware" / "MRTanh0.18.pth",
        "architecture": "MirroredResNet",
        "activation": "tanh",
        "alpha": 0.18,
        "accuracy": 0.7855,
        "ce_loss": 1.2080,
        "compaction_pct": 0.94,
        "description": "Mirrored ResNet with Tanh - lowest CE loss"
    },
    "MR Sigmoid": {
        "path": MODELS_PATH / "Task-Aware" / "MRSigmoid0.00.pth",
        "architecture": "MirroredResNet",
        "activation": "sigmoid",
        "alpha": 0.00,
        "accuracy": 0.7860,
        "ce_loss": 1.5268,
        "compaction_pct": -10.92,
        "description": "Mirrored ResNet with Sigmoid - most fidelity"
    }
}

# ═══════════════════════════════════════════════════════════════════════════════
# 🎨 MULTICLASS DECODER CONFIGURATIONS
# ═══════════════════════════════════════════════════════════════════════════════

MULTICLASS_DECODERS: Dict[str, Dict] = {
    "Simple Tanh": {
        "path": MULTICLASS_MODELS_PATH / "SimpleTanh_AccCE" / "decoder_alpha0.01_seed43.pth",
        "architecture": "Simple",
        "activation": "tanh",
        "alpha": 0.01,
        "accuracy": 0.7840,
        "ce_loss": 0.0026,
        "compaction_pct": 12.725,
        "description": "Simple Tanh - CIFAR-10 optimized (Accuracy-to-CE)"
    },
    "Simple Sigmoid": {
        "path": MULTICLASS_MODELS_PATH / "SimpleSigmoid_Acc" / "decoder_alpha0.20_seed43.pth",
        "architecture": "Simple",
        "activation": "sigmoid",
        "alpha": 0.20,
        "accuracy": 0.7835,
        "ce_loss": 0.0013,
        "compaction_pct": 12.697,
        "description": "Simple Sigmoid - CIFAR-10 optimized (Peak Accuracy)"
    },
    "MR Tanh": {
        "path": MULTICLASS_MODELS_PATH / "MRTanh_AccCE" / "decoder_alpha0.18_seed43.pth",
        "architecture": "MirroredResNet",
        "activation": "tanh",
        "alpha": 0.18,
        "accuracy": 0.7818,
        "ce_loss": 0.0057,
        "compaction_pct": 14.322,
        "description": "MR Tanh - CIFAR-10 optimized (Accuracy-to-CE)"
    },
    "MR Sigmoid": {
        "path": MULTICLASS_MODELS_PATH / "MRSigmoid_Comp" / "decoder_alpha0.32_seed43.pth",
        "architecture": "MirroredResNet",
        "activation": "sigmoid",
        "alpha": 0.32,
        "accuracy": 0.7813,
        "ce_loss": 0.0041,
        "compaction_pct": 13.515,
        "description": "MR Sigmoid - CIFAR-10 optimized (Max Compaction)"
    }
}

# ═══════════════════════════════════════════════════════════════════════════════
# 👤 CLASSIFIER CONFIGURATIONS
# ═══════════════════════════════════════════════════════════════════════════════

BINARY_CLASSIFIER: Dict = {
    "path": CLASSIFIERS_PATH / "binary_classifier.pth",
    "accuracy": 0.815,
    "ce_loss": 1.5279,
    "classes": BINARY_CLASSES,
    "num_classes": 2,
    "description": "Binary classifier trained on Cats vs Dogs"
}

MULTICLASS_CLASSIFIER: Dict = {
    "path": CLASSIFIERS_PATH / "multiclass_classifier.pth",
    "accuracy": 0.7771,
    "ce_loss": 1.0591,
    "classes": CIFAR10_CLASSES,
    "num_classes": 10,
    "description": "Multiclass classifier trained on full CIFAR-10"
}

# ═══════════════════════════════════════════════════════════════════════════════
# 📈 BASELINE COMPARISONS
# ═══════════════════════════════════════════════════════════════════════════════

BASELINES: Dict[str, Dict] = {
    "Deep Autoencoder": {
        "accuracy": 0.6937,
        "ce_loss": 1.1514,
        "compaction_pct": 8.34,
        "description": "Reconstruction-only CNN autoencoder (no task awareness)",
        "color": "#FF6B6B"
    },
    "Lossy JPEG": {
        "accuracy": 0.45,
        "ce_loss": 2.2292,
        "compaction_pct": 32.71,
        "description": "Standard lossy JPEG compression at matched file size",
        "color": "#FFA500"
    },
    "Baseline (Full Images)": {
        "accuracy": 0.7771,
        "ce_loss": 1.0591,
        "compaction_pct": 0.0,
        "description": "Full CIFAR-10 images without compression",
        "color": "#4ECDC4"
    },
    "Baseline Binary": {
        "accuracy": 0.815,
        "ce_loss": 1.5279,
        "compaction_pct": 0.0,
        "description": "Full Cat/Dog images without compression",
        "color": "#95E1D3"
    }
}

# ═══════════════════════════════════════════════════════════════════════════════
# 📊 EXPERIMENT METRICS
# ═══════════════════════════════════════════════════════════════════════════════

BINARY_METRICS: Dict = {
    "total_experiments": 404,
    "alpha_range": "0.00 to 1.00",
    "alpha_increments": 0.01,
    "architectures": 4,
    "epochs_per_decoder": 50,
    "total_gpu_hours": 28,
    "gpu_model": "NVIDIA RTX 3050 (4GB VRAM)"
}

MULTICLASS_METRICS: Dict = {
    "total_experiments": 60,
    "alpha_values": 3,
    "architectures": 4,
    "random_seeds": [2, 19, 43, 127, 409],
    "epochs_per_decoder": 50,
    "total_gpu_hours": 24,
    "gpu_model": "NVIDIA RTX 3050 (4GB/6GB VRAM)"
}

# ═══════════════════════════════════════════════════════════════════════════════
# 🔐 EXTRACTOR SPECIFICITY (Privacy Analysis)
# ═══════════════════════════════════════════════════════════════════════════════

EXTRACTOR_SPECIFICITY: Dict = {
    "trained_on_original": {
        "tested_on_original": 0.8207,
        "tested_on_reconstructions": 0.6839
    },
    "trained_on_reconstructions": {
        "tested_on_original": 0.1123,
        "tested_on_reconstructions": 0.7807
    },
    "cross_model_transfer": 0.1123,
    "interpretation": "Only 11.23% transfer when trained on reconstructions tested on originals - strong privacy preservation"
}

# ═══════════════════════════════════════════════════════════════════════════════
# 🛠️ HELPER FUNCTIONS - MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def get_decoders(mode: str = None) -> Dict[str, Dict]:
    """
    Get decoder configurations based on load mode.
    
    Args:
        mode (str, optional): Override LOAD_MODE. "binary" or "multiclass"
    
    Returns:
        Dict[str, Dict]: Decoder configurations
    
    Example:
        decoders = get_decoders()  # Uses LOAD_MODE
        decoders = get_decoders("binary")  # Force binary
    """
    load_mode = mode or LOAD_MODE
    
    if load_mode == "binary":
        return BINARY_DECODERS
    elif load_mode == "multiclass":
        return MULTICLASS_DECODERS
    else:
        raise ValueError(f"❌ Invalid LOAD_MODE: {load_mode}. Use 'binary' or 'multiclass'")


def get_classifier(mode: str = None) -> Dict:
    """
    Get classifier configuration based on load mode.
    
    Args:
        mode (str, optional): Override LOAD_MODE. "binary" or "multiclass"
    
    Returns:
        Dict: Classifier configuration with path and metadata
    """
    load_mode = mode or LOAD_MODE
    
    if load_mode == "binary":
        return BINARY_CLASSIFIER
    elif load_mode == "multiclass":
        return MULTICLASS_CLASSIFIER
    else:
        raise ValueError(f"❌ Invalid LOAD_MODE: {load_mode}. Use 'binary' or 'multiclass'")


def get_classes(mode: str = None) -> Dict[int, str]:
    """
    Get class mapping based on load mode.
    
    Args:
        mode (str, optional): Override LOAD_MODE. "binary" or "multiclass"
    
    Returns:
        Dict[int, str]: Class index to name mapping
    """
    load_mode = mode or LOAD_MODE
    
    if load_mode == "binary":
        return BINARY_CLASSES
    elif load_mode == "multiclass":
        return CIFAR10_CLASSES
    else:
        raise ValueError(f"❌ Invalid LOAD_MODE: {load_mode}. Use 'binary' or 'multiclass'")


def get_baseline(metric_type: str = "multiclass") -> Dict:
    """
    Get baseline metrics for comparison.
    
    Args:
        metric_type (str): "binary" or "multiclass"
    
    Returns:
        Dict: Baseline metrics
    """
    if metric_type == "binary":
        return BASELINES["Baseline Binary"]
    elif metric_type == "multiclass":
        return BASELINES["Baseline (Full Images)"]
    else:
        raise ValueError(f"❌ Invalid metric_type: {metric_type}")


def get_mode_info() -> Dict:
    """
    Get comprehensive information about current load mode.
    
    Returns:
        Dict: Mode information including decoders, classifier, metrics
    """
    decoders = get_decoders()
    classifier = get_classifier()
    classes = get_classes()
    
    return {
        "mode": LOAD_MODE,
        "num_decoders": len(decoders),
        "decoder_names": list(decoders.keys()),
        "num_classes": len(classes),
        "class_names": list(classes.values()),
        "classifier_accuracy": classifier["accuracy"],
        "is_binary": LOAD_MODE == "binary",
        "is_multiclass": LOAD_MODE == "multiclass"
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 🧪 SELF-TEST (Run with: python config.py)
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "="*80)
    print("  CONFIG.PY - Configuration Test & Information")
    print("="*80 + "\n")
    
    mode_info = get_mode_info()
    
    print(f"🎯 Current Mode: {LOAD_MODE.upper()}")
    print(f"📦 Decoders: {mode_info['num_decoders']} ({', '.join(mode_info['decoder_names'])})")
    print(f"🏷️  Classes: {mode_info['num_classes']} ({', '.join(mode_info['class_names'])})")
    print(f"📊 Baseline Accuracy: {mode_info['classifier_accuracy']:.1%}")
    
    print("\n" + "-"*80)
    print("Available Decoders:")
    print("-"*80)
    
    decoders = get_decoders()
    for name, config in decoders.items():
        print(f"\n  {name}")
        print(f"    Architecture: {config['architecture']}")
        print(f"    Activation: {config['activation']}")
        print(f"    Alpha (α): {config['alpha']}")
        print(f"    Accuracy: {config['accuracy']:.1%}")
        print(f"    Compaction: {config['compaction_pct']:+.2f}%")
    
    print("\n" + "="*80)
    print("✅ CONFIG.PY loaded successfully!")
    print("="*80 + "\n")
