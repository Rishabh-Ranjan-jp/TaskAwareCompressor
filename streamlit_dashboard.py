"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                      STREAMLIT_DASHBOARD.PY                                 ║
║         Task-Aware Image Reconstruction - Interactive Dashboard             ║
║                  PREMIUM DESIGN + COMPREHENSIVE BINARY ANALYSIS             ║
║                                                                              ║
║  Smart model loading:                                                       ║
║  - Analysis/Graphs: ALWAYS uses multiclass (full CIFAR-10)                 ║
║  - Interactive Demo: ALWAYS uses binary (custom images only)               ║
║  - NEW: Comprehensive binary decoder analysis + baselines                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from torchvision import models, transforms
from torchvision.models import resnet18, ResNet18_Weights
import os

# ───────────────────────────────────────────────────────────────────────────────
# 📥 IMPORTS FROM CONFIG
# ───────────────────────────────────────────────────────────────────────────────

from config import (
    CUSTOM_DATASET_PATH, CIFAR10_MEAN, CIFAR10_STD,
    BINARY_METRICS, MULTICLASS_METRICS, BASELINES, EXTRACTOR_SPECIFICITY,
    BINARY_CLASSES, CIFAR10_CLASSES,
    BINARY_DECODERS, MULTICLASS_DECODERS,
    BINARY_CLASSIFIER, MULTICLASS_CLASSIFIER
)

# ═══════════════════════════════════════════════════════════════════════════════
# 🎨 PAGE CONFIGURATION & STYLING - PREMIUM DESIGN
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Task-Aware Image Reconstruction",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium CSS styling
st.markdown("""
<style>
    :root {
        --primary-color: #1E3A8A;
        --secondary-color: #F59E0B;
        --success-color: #10B981;
        --danger-color: #EF4444;
        --light-bg: #F8FAFC;
        --card-shadow: 0 10px 25px rgba(0, 0, 0, 0.08);
    }
    
    * {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 24px;
        border-radius: 12px;
        color: white;
        margin: 10px 0;
        box-shadow: var(--card-shadow);
        border: 1px solid rgba(255,255,255,0.1);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.12);
    }
    
    .premium-header {
        background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 100%);
        color: white;
        padding: 40px 20px;
        border-radius: 12px;
        margin-bottom: 30px;
        box-shadow: var(--card-shadow);
    }
    
    .section-header {
        color: #1E3A8A;
        font-weight: 700;
        font-size: 24px;
        margin-top: 30px;
        margin-bottom: 20px;
        border-left: 4px solid #F59E0B;
        padding-left: 15px;
    }
    
    .decoder-title {
        color: #1E3A8A;
        font-weight: 700;
        font-size: 18px;
    }
    
    .mode-badge {
        padding: 10px 18px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        margin: 5px 0;
        font-size: 14px;
        letter-spacing: 0.5px;
    }
    
    .mode-badge-binary {
        background: linear-gradient(135deg, #E0F2FE 0%, #BAE6FD 100%);
        color: #0C4A6E;
        border: 1px solid #0EA5E9;
    }
    
    .mode-badge-multiclass {
        background: linear-gradient(135deg, #FEF3C7 0%, #FCD34D 100%);
        color: #92400E;
        border: 1px solid #F59E0B;
    }
    
    .info-card {
        background: linear-gradient(135deg, #F0F9FF 0%, #E0F2FE 100%);
        padding: 20px;
        border-radius: 12px;
        border-left: 4px solid #0EA5E9;
        margin: 15px 0;
    }
    
    .success-card {
        background: linear-gradient(135deg, #F0FDF4 0%, #DBEAFE 100%);
        padding: 20px;
        border-radius: 12px;
        border-left: 4px solid #10B981;
    }
    
    .st-dataframe {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: var(--card-shadow);
    }
    
    .stMetric {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        border: 1px solid #E5E7EB;
    }
    
    .stPlotlyChart {
        border-radius: 12px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 🏗️ ARCHITECTURE CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

class ResNet18Encoder(nn.Module):
    """ResNet-18 encoder up to Layer 3 (256 channels, 8×8 spatial)"""
    def __init__(self):
        super().__init__()
        base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.features = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,
            base_model.layer1,
            base_model.layer2,
            base_model.layer3
        )
    
    def forward(self, x):
        return self.features(x)


class SimpleDecoder(nn.Module):
    """Simple decoder: 4 ConvTranspose layers (256→128→64→32→3)"""
    def __init__(self, activation='tanh'):
        super().__init__()
        self.decode = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh() if activation == 'tanh' else nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.decode(x)


class MirrorBasicBlock(nn.Module):
    """Mirrored ResNet BasicBlock with residual connections"""
    def __init__(self, block):
        super().__init__()
        self.deconv2 = self._mirror_conv(block.conv2)
        self.bn2 = nn.BatchNorm2d(block.conv2.in_channels)
        self.deconv1 = self._mirror_conv(block.conv1)
        self.bn1 = nn.BatchNorm2d(block.conv1.in_channels)
        self.relu = nn.ReLU(inplace=True)
        
        if block.downsample is not None:
            down = block.downsample[0]
            self.upsample = nn.Sequential(
                self._mirror_conv(down),
                nn.BatchNorm2d(down.in_channels)
            )
        else:
            self.upsample = None
    
    def _mirror_conv(self, conv):
        return nn.ConvTranspose2d(
            in_channels=conv.out_channels,
            out_channels=conv.in_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            output_padding=1 if conv.stride[0] > 1 else 0,
            bias=False
        )
    
    def forward(self, x):
        identity = x
        out = self.deconv2(x)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.deconv1(out)
        out = self.bn1(out)
        if self.upsample is not None:
            identity = self.upsample(identity)
        out += identity
        out = self.relu(out)
        return out


class MirroredResNetDecoder(nn.Module):
    """Fully mirrored ResNet decoder with residual connections"""
    def __init__(self, base_model=None, activation='tanh'):
        super().__init__()
        if base_model is None:
            base_model = models.resnet18(weights=None)
        
        def mirror_layer(layer):
            blocks = [MirrorBasicBlock(b) for b in reversed(layer)]
            return nn.Sequential(*blocks)
        
        self.up3 = mirror_layer(base_model.layer3)
        self.up2 = mirror_layer(base_model.layer2)
        self.up1 = mirror_layer(base_model.layer1)
        self.final = nn.ConvTranspose2d(64, 3, kernel_size=7, stride=2, padding=3, output_padding=1)
        
        final_activation = nn.Tanh() if activation == 'tanh' else nn.Sigmoid()
        self.upsample_final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1),
            final_activation
        )
    
    def forward(self, x):
        x = self.up3(x)
        x = self.up2(x)
        x = self.up1(x)
        x = self.final(x)
        x = self.upsample_final(x)
        return x


class DeepAutoencoder(nn.Module):
    """Deep Autoencoder for baseline comparison"""
    def __init__(self):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.ReLU(True),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class BinaryClassifier(nn.Module):
    """Binary classifier wrapper (Cats vs Dogs)"""
    def __init__(self):
        super().__init__()
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 2)
        
        for param in self.model.parameters():
            param.requires_grad = False
        
        for layer in [self.model.layer3, self.model.layer4, self.model.fc]:
            for param in layer.parameters():
                param.requires_grad = True
    
    def forward(self, x):
        return self.model(x)


class MulticlassClassifier(nn.Module):
    """Multiclass classifier wrapper (CIFAR-10)"""
    def __init__(self):
        super().__init__()
        base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.features = nn.Sequential(*list(base_model.children())[:-1])
        self.fc = nn.Linear(512, 10)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# ═══════════════════════════════════════════════════════════════════════════════
# 💾 CACHED MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_analysis_models():
    """Load MULTICLASS models for analysis/graphs/results"""
    models_dict = {}
    
    with st.spinner("🔄 Loading multiclass models for analysis..."):
        encoder = ResNet18Encoder()
        encoder.eval()
        models_dict['encoder'] = encoder
        
        classifier = MulticlassClassifier()
        classifier.eval()
        models_dict['classifier'] = classifier
        
        models_dict['decoders'] = {}
        for decoder_name, decoder_info in MULTICLASS_DECODERS.items():
            decoder_class = SimpleDecoder if decoder_info['architecture'] == "Simple" else MirroredResNetDecoder
            decoder = decoder_class(activation=decoder_info['activation'])
            decoder.eval()
            
            try:
                state_dict = torch.load(decoder_info['path'], map_location='cpu', weights_only=True)
                decoder.load_state_dict(state_dict)
                models_dict['decoders'][decoder_name] = decoder
            except Exception as e:
                st.warning(f"⚠️ Could not load {decoder_name}: {str(e)}")
    
    return models_dict


@st.cache_resource
def load_demo_models():
    """Load BINARY models for interactive demo"""
    models_dict = {}
    
    with st.spinner("🔄 Loading binary models for demo..."):
        encoder = ResNet18Encoder()
        encoder.eval()
        models_dict['encoder'] = encoder
        
        classifier = BinaryClassifier()
        classifier.eval()
        models_dict['classifier'] = classifier
        
        # Deep Autoencoder for baseline
        deep_ae = DeepAutoencoder()
        deep_ae.eval()
        models_dict['deep_ae'] = deep_ae
        
        models_dict['decoders'] = {}
        for decoder_name, decoder_info in BINARY_DECODERS.items():
            decoder_class = SimpleDecoder if decoder_info['architecture'] == "Simple" else MirroredResNetDecoder
            decoder = decoder_class(activation=decoder_info['activation'])
            decoder.eval()
            
            try:
                state_dict = torch.load(decoder_info['path'], map_location='cpu', weights_only=True)
                decoder.load_state_dict(state_dict)
                models_dict['decoders'][decoder_name] = decoder
            except Exception as e:
                st.warning(f"⚠️ Could not load {decoder_name}: {str(e)}")
    
    return models_dict

# ═══════════════════════════════════════════════════════════════════════════════
# 🖼️ IMAGE PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Convert PIL image to normalized tensor"""
    image = image.resize((32, 32))
    image_array = np.array(image) / 255.0
    image_tensor = torch.FloatTensor(image_array).permute(2, 0, 1).unsqueeze(0)
    
    normalize = transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
    image_tensor = normalize(image_tensor)
    
    return image_tensor


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """Convert tensor back to PIL image"""
    tensor = tensor.squeeze(0).detach().cpu()
    
    for c in range(3):
        tensor[c] = tensor[c] * CIFAR10_STD[c] + CIFAR10_MEAN[c]
    
    tensor = torch.clamp(tensor, 0, 1)
    image_array = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(image_array)

# ═══════════════════════════════════════════════════════════════════════════════
# 📄 PAGE: HOME & OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════

def page_home():
    """Home page with project overview"""
    st.markdown("""
    <div class="premium-header">
        <h1 style="margin: 0; font-size: 42px; font-weight: 800;">🖼️ Task-Aware Image Reconstruction</h1>
        <p style="margin: 10px 0 0 0; font-size: 18px; opacity: 0.95;">Normalized Loss Balancing for Efficient Vision Systems</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Hero section
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 The Problem
        
        Traditional image compression:
        - ❌ Optimized for human perception
        - ❌ Degrades classification accuracy
        - ❌ Wastes bandwidth on irrelevant details
        - ❌ No task awareness
        """)
    
    with col2:
        st.markdown("""
        ### ✅ Our Solution
        
        **Normalized Loss Balancing**:
        - ✓ Task-aware compression
        - ✓ Stable multi-objective optimization
        - ✓ Interpretable α ∈ [0,1] control
        - ✓ Exponential Moving Average normalization
        """)
    
    st.markdown("---")
    st.markdown("<h2 class='section-header'>💡 The Key Innovation</h2>", unsafe_allow_html=True)
    
    innovation_cols = st.columns(3)
    
    with innovation_cols[0]:
        st.markdown("""
        #### ❌ Fixed Weights
        ```
        Loss = α·CE + (1-α)·MSE
        
        CE: 0-3
        MSE: 10³-10⁵
        
        ❌ Imbalanced scales
        ❌ α meaningless
        ❌ Unstable
        ```
        """)
    
    with innovation_cols[1]:
        st.markdown("""
        #### 🔄 EMA Normalization
        ```
        μ_CE = EMA(CE_t)
        μ_MSE = EMA(MSE_t)
        
        Loss = α·CE/μ_CE + 
               (1-α)·MSE/μ_MSE
        
        ✓ Balanced scales
        ✓ α meaningful
        ✓ Stable
        ```
        """)
    
    with innovation_cols[2]:
        st.markdown("""
        #### ✅ Results
        ```
        Interpretable control:
        
        α=0: Max compression
        α=0.5: Balanced
        α=1: Full fidelity
        
        Consistent across
        all 4 architectures!
        ```
        """)
    
    st.markdown("---")
    st.markdown("<h2 class='section-header'>📊 Key Results</h2>", unsafe_allow_html=True)
    
    res_col1, res_col2, res_col3 = st.columns(3)
    
    with res_col1:
        st.metric(
            label="🎯 Accuracy Maintained",
            value="78.40%",
            delta="+0.69%"
        )
        st.caption("vs baseline 77.71%")
    
    with res_col2:
        st.metric(
            label="📦 Average Compaction",
            value="13.29%",
            delta="File size reduction"
        )
        st.caption("15-23% per decoder")
    
    with res_col3:
        st.metric(
            label="🔐 Privacy Transfer",
            value="11.23%",
            delta="Cross-model only"
        )
        st.caption("Strong preservation")
    
    st.markdown("---")
    st.markdown("<h2 class='section-header'>🔬 Experimental Scale</h2>", unsafe_allow_html=True)
    
    exp_col1, exp_col2, exp_col3 = st.columns(3)
    
    with exp_col1:
        st.markdown(f"""
        #### 📈 Multiclass (CIFAR-10)
        - **{MULTICLASS_METRICS['total_experiments']}** experiments
        - **{MULTICLASS_METRICS['architectures']}** architectures
        - **{MULTICLASS_METRICS['total_gpu_hours']}** GPU hours
        - **10** classes
        """)
    
    with exp_col2:
        st.markdown(f"""
        #### 👥 Binary (Custom Dataset)
        - **{BINARY_METRICS['total_experiments']}** experiments
        - **{BINARY_METRICS['architectures']}** architectures
        - **{BINARY_METRICS['total_gpu_hours']}** GPU hours
        - **2** classes
        """)
    
    with exp_col3:
        st.markdown(f"""
        #### 💻 Hardware
        - **GPU:** RTX 3050 (4GB/6GB)
        - **CPU:** Intel i5/i7
        - **Total:** **{BINARY_METRICS['total_gpu_hours'] + MULTICLASS_METRICS['total_gpu_hours']}** GPU hours
        - **Data:** CIFAR-10 + Custom
        """)

# ═══════════════════════════════════════════════════════════════════════════════
# 📈 PAGE: RESULTS DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

def page_results():
    """Results and performance analysis"""
    st.title("📈 Results Dashboard")
    
    st.markdown("""
    <div class="mode-badge mode-badge-multiclass">📊 Showing Multiclass Results (CIFAR-10)</div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"**Experiments**: {MULTICLASS_METRICS['total_experiments']} | **Architectures**: {MULTICLASS_METRICS['architectures']} | **GPU Hours**: {MULTICLASS_METRICS['total_gpu_hours']}")
    
    st.markdown("---")
    st.markdown("<h2 class='section-header'>🎨 Decoder Performance</h2>", unsafe_allow_html=True)
    
    decoder_data = []
    for name, config in MULTICLASS_DECODERS.items():
        decoder_data.append({
            "Decoder": name,
            "Architecture": config['architecture'],
            "Activation": config['activation'].title(),
            "α": config['alpha'],
            "Accuracy": config['accuracy'],
            "CE Loss": config['ce_loss'],
            "Compaction %": config['compaction_pct']
        })
    
    df_decoders = pd.DataFrame(decoder_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Accuracy by Decoder")
        fig1 = px.bar(df_decoders, x='Decoder', y='Accuracy', 
                      color='Architecture', title='Classification Accuracy',
                      color_discrete_sequence=['#3B82F6', '#8B5CF6'])
        fig1.update_traces(marker_line_width=2, marker_line_color='white')
        fig1.update_layout(showlegend=True, hovermode='x unified')
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.subheader("Compaction by Decoder")
        fig2 = px.bar(df_decoders, x='Decoder', y='Compaction %', 
                      color='Activation', title='File Size Compaction',
                      color_discrete_sequence=['#F59E0B', '#10B981'])
        fig2.update_traces(marker_line_width=2, marker_line_color='white')
        fig2.update_layout(showlegend=True, hovermode='x unified')
        st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("---")
    st.markdown("<h2 class='section-header'>📊 vs Baselines</h2>", unsafe_allow_html=True)
    
    baseline_data = []
    for name, config in BASELINES.items():
        baseline_data.append({
            "Method": name,
            "Accuracy": config['accuracy'] * 100,
            "Compaction %": config['compaction_pct'],
            "Size": 60,
            "Type": "Baseline"
        })
    
    for name, config in MULTICLASS_DECODERS.items():
        baseline_data.append({
            "Method": f"Ours: {name}",
            "Accuracy": config['accuracy'] * 100,
            "Compaction %": config['compaction_pct'],
            "Size": 80,
            "Type": "Our Method"
        })
    
    df_baseline = pd.DataFrame(baseline_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Accuracy vs Compaction")
        fig3 = px.scatter(df_baseline, x='Compaction %', y='Accuracy',
                         hover_name='Method', size='Size',
                         color='Type',
                         title='Trade-off Analysis',
                         color_discrete_map={'Baseline': '#EF4444', 'Our Method': '#10B981'})
        fig3.update_layout(height=400, hovermode='closest')
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        st.subheader("CE Loss (Log Scale)")
        loss_data = []
        for name, config in BASELINES.items():
            loss_data.append({
                "Method": name,
                "CE Loss": config['ce_loss'],
                "Type": "Baseline",
                "Log Loss": np.log10(config['ce_loss'] + 1e-6)
            })
        
        for name, config in MULTICLASS_DECODERS.items():
            loss_data.append({
                "Method": f"Ours: {name}",
                "CE Loss": config['ce_loss'],
                "Type": "Our Method",
                "Log Loss": np.log10(config['ce_loss'] + 1e-6)
            })
        
        df_loss = pd.DataFrame(loss_data)
        fig4 = px.bar(df_loss, x='Method', y='Log Loss',
                     title='Cross-Entropy Loss (Log Scale)',
                     color='Type',
                     color_discrete_map={'Baseline': '#EF4444', 'Our Method': '#10B981'})
        fig4.update_layout(height=400)
        st.plotly_chart(fig4, use_container_width=True)
    
    st.markdown("---")
    st.markdown("<h2 class='section-header'>📋 Detailed Metrics</h2>", unsafe_allow_html=True)
    st.dataframe(df_decoders.style.format({
        'Accuracy': '{:.1%}',
        'CE Loss': '{:.4f}',
        'Compaction %': '{:+.2f}',
        'α': '{:.2f}'
    }), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 🔬 PAGE: INTERACTIVE DEMO + BINARY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def page_demo():
    """Interactive demo with comprehensive binary analysis"""
    st.title("🔬 Interactive Decoder Demonstration & Analysis")
    
    st.markdown("""
    <div class="mode-badge mode-badge-binary">🎯 Binary Demo Mode (Cats vs Dogs)</div>
    """, unsafe_allow_html=True)
    
    # Choose section
    demo_section = st.radio(
        "Select Section:",
        ["📊 Comprehensive Binary Analysis", "🎨 Interactive Single Decoder Demo"],
        horizontal=True
    )
    
    models_dict = load_demo_models()
    
    # ═════════════════════════════════════════════════════════════════════════════
    # SECTION 1: COMPREHENSIVE BINARY ANALYSIS
    # ═════════════════════════════════════════════════════════════════════════════
    
    if demo_section == "📊 Comprehensive Binary Analysis":
        st.markdown("<h2 class='section-header'>Full Binary Decoder Analysis on Custom Dataset</h2>", unsafe_allow_html=True)
        
        st.info("""
        This section runs ALL binary decoders on your custom dataset and:
        - Measures accuracy preservation
        - Calculates compression rates
        - Compares with Deep Autoencoder baseline
        - Reports baseline performance
        """)
        
        # Load custom dataset
        custom_files = sorted([f for f in CUSTOM_DATASET_PATH.glob("*") if f.is_file()])
        
        if not custom_files:
            st.warning("❌ No images found in custom_dataset/")
            return
        
        if st.button("🚀 Run Comprehensive Analysis", type="primary", use_container_width=True):
            with st.spinner("Analyzing all decoders on custom dataset..."):
                # Prepare data collection
                results = []
                
                # BASELINE 1: Raw images (no decoder)
                st.subheader("📌 Phase 1: Baseline - Raw Images (No Decoder)")
                baseline_correct = 0
                baseline_total = len(custom_files)
                
                progress_bar = st.progress(0)
                for idx, img_path in enumerate(custom_files):
                    try:
                        img = Image.open(img_path).convert('RGB')
                        img_tensor = preprocess_image(img)
                        
                        with torch.no_grad():
                            pred = models_dict['classifier'](img_tensor)
                            # True label from filename
                            true_label = "Cat" if "cat" in img_path.name.lower() else "Dog"
                            pred_label = "Cat" if pred.argmax().item() == 0 else "Dog"
                            
                            if true_label == pred_label:
                                baseline_correct += 1
                    except:
                        pass
                    
                    progress_bar.progress((idx + 1) / baseline_total)
                
                baseline_acc = baseline_correct / baseline_total if baseline_total > 0 else 0
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("✅ Baseline Accuracy", f"{baseline_acc:.1%}", f"{baseline_correct}/{baseline_total} correct")
                with col2:
                    st.metric("📊 Total Images", baseline_total)
                with col3:
                    st.metric("🎯 CE Loss", "0.0", "No reconstruction")
                
                # BASELINE 2: Deep Autoencoder
                st.subheader("📌 Phase 2: Deep Autoencoder Baseline")
                deep_ae_correct = 0
                deep_ae_total = len(custom_files)
                
                progress_bar = st.progress(0)
                for idx, img_path in enumerate(custom_files):
                    try:
                        img = Image.open(img_path).convert('RGB')
                        img_tensor = preprocess_image(img)
                        
                        with torch.no_grad():
                            recon = models_dict['deep_ae'](img_tensor)
                            pred = models_dict['classifier'](recon)
                            
                            true_label = "Cat" if "cat" in img_path.name.lower() else "Dog"
                            pred_label = "Cat" if pred.argmax().item() == 0 else "Dog"
                            
                            if true_label == pred_label:
                                deep_ae_correct += 1
                    except:
                        pass
                    
                    progress_bar.progress((idx + 1) / deep_ae_total)
                
                deep_ae_acc = deep_ae_correct / deep_ae_total if deep_ae_total > 0 else 0
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("✅ Deep AE Accuracy", f"{deep_ae_acc:.1%}", f"{deep_ae_correct}/{deep_ae_total} correct")
                with col2:
                    st.metric("📊 Total Images", deep_ae_total)
                with col3:
                    st.metric("📉 Drop", f"{(baseline_acc - deep_ae_acc):.1%}", "from baseline")
                
                results.append({
                    "Method": "Baseline (Raw Images)",
                    "Type": "Baseline",
                    "Accuracy": baseline_acc,
                    "CE Loss": 0.0,
                    "Compaction %": 0.0,
                    "Images Tested": baseline_total,
                    "Correct": baseline_correct
                })
                
                results.append({
                    "Method": "Deep Autoencoder",
                    "Type": "Baseline",
                    "Accuracy": deep_ae_acc,
                    "CE Loss": 0.0,
                    "Compaction %": 8.34,
                    "Images Tested": deep_ae_total,
                    "Correct": deep_ae_correct
                })
                
                # PHASE 3: Binary Decoders
                st.subheader("📌 Phase 3: Task-Aware Binary Decoders")
                
                for decoder_name, decoder_info in BINARY_DECODERS.items():
                    st.write(f"**Testing {decoder_name}...**")
                    decoder_correct = 0
                    decoder_total = len(custom_files)
                    
                    progress_bar = st.progress(0)
                    for idx, img_path in enumerate(custom_files):
                        try:
                            img = Image.open(img_path).convert('RGB')
                            img_tensor = preprocess_image(img)
                            
                            with torch.no_grad():
                                features = models_dict['encoder'](img_tensor)
                                recon = models_dict['decoders'][decoder_name](features)
                                pred = models_dict['classifier'](recon)
                                
                                true_label = "Cat" if "cat" in img_path.name.lower() else "Dog"
                                pred_label = "Cat" if pred.argmax().item() == 0 else "Dog"
                                
                                if true_label == pred_label:
                                    decoder_correct += 1
                        except:
                            pass
                        
                        progress_bar.progress((idx + 1) / decoder_total)
                    
                    decoder_acc = decoder_correct / decoder_total if decoder_total > 0 else 0
                    
                    results.append({
                        "Method": decoder_name,
                        "Type": "Our Method",
                        "Accuracy": decoder_acc,
                        "CE Loss": decoder_info['ce_loss'],
                        "Compaction %": decoder_info['compaction_pct'],
                        "Images Tested": decoder_total,
                        "Correct": decoder_correct
                    })
                
                # Display comprehensive results
                st.markdown("---")
                st.markdown("<h2 class='section-header'>📊 Comprehensive Results</h2>", unsafe_allow_html=True)
                
                df_results = pd.DataFrame(results)
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                
                best_decoder = df_results[df_results['Type'] == 'Our Method'].nlargest(1, 'Accuracy').iloc[0]
                
                with col1:
                    st.metric("🏆 Best Decoder Accuracy", f"{best_decoder['Accuracy']:.1%}", best_decoder['Method'])
                
                with col2:
                    st.metric("📦 Best Compaction", f"{best_decoder['Compaction %']:.2f}%", best_decoder['Method'])
                
                with col3:
                    acc_drop = baseline_acc - best_decoder['Accuracy']
                    st.metric("📉 Accuracy Drop vs Raw", f"{acc_drop:.1%}", "minimal loss")
                
                with col4:
                    deep_ae_vs_ours = best_decoder['Accuracy'] - deep_ae_acc
                    st.metric("✅ Improvement vs Deep AE", f"{deep_ae_vs_ours:+.1%}", f"{deep_ae_vs_ours*baseline_total:.0f} more correct")
                
                # Visualizations
                st.markdown("---")
                
                viz_col1, viz_col2 = st.columns(2)
                
                with viz_col1:
                    fig_acc = px.bar(df_results, x='Method', y='Accuracy',
                                    color='Type',
                                    title='Accuracy Across All Methods',
                                    color_discrete_map={'Baseline': '#EF4444', 'Our Method': '#10B981'},
                                    labels={'Method': 'Decoder', 'Accuracy': 'Accuracy (%)'})
                    fig_acc.update_layout(height=400)
                    st.plotly_chart(fig_acc, use_container_width=True)
                
                with viz_col2:
                    fig_comp = px.bar(df_results, x='Method', y='Compaction %',
                                     color='Type',
                                     title='Compaction % Across Methods',
                                     color_discrete_map={'Baseline': '#9CA3AF', 'Our Method': '#F59E0B'},
                                     labels={'Method': 'Decoder', 'Compaction %': 'Compaction (%)'})
                    fig_comp.update_layout(height=400)
                    st.plotly_chart(fig_comp, use_container_width=True)
                
                # Detailed table
                st.markdown("---")
                st.subheader("📋 Detailed Results Table")
                st.dataframe(df_results.style.format({
                    'Accuracy': '{:.1%}',
                    'CE Loss': '{:.4f}',
                    'Compaction %': '{:+.2f}'
                }), use_container_width=True)
                
                st.markdown("""
                <div class="success-card">
                ✅ Analysis Complete!
                
                **Key Findings:**
                - Raw images: {:.1%} accuracy
                - Deep Autoencoder: {:.1%} accuracy ({:+.1%} vs raw)
                - Best task-aware decoder: {:.1%} accuracy ({:+.1%} vs raw)
                </div>
                """.format(baseline_acc, deep_ae_acc, deep_ae_acc - baseline_acc, 
                          best_decoder['Accuracy'], best_decoder['Accuracy'] - baseline_acc), 
                unsafe_allow_html=True)
    
    # ═════════════════════════════════════════════════════════════════════════════
    # SECTION 2: INTERACTIVE SINGLE DECODER
    # ═════════════════════════════════════════════════════════════════════════════
    
    else:
        st.markdown("<h2 class='section-header'>Single Decoder Interactive Demo</h2>", unsafe_allow_html=True)
        
        st.info("""
        Upload an image and choose a decoder to see real-time reconstruction.
        """)
        
        st.subheader("📸 Step 1: Select Image")
        
        source = st.radio("Choose image source:", ["Upload Image", "Use Sample from Dataset"])
        
        image = None
        if source == "Upload Image":
            uploaded = st.file_uploader("Choose an image (JPG/PNG)", type=['jpg', 'png', 'jpeg'])
            if uploaded:
                image = Image.open(uploaded)
        else:
            custom_files = sorted([f for f in CUSTOM_DATASET_PATH.glob("*") if f.is_file()])
            if custom_files:
                selected = st.selectbox("Select from dataset:", custom_files)
                image = Image.open(selected)
            else:
                st.warning("❌ No images found in custom_dataset/")
        
        if image is None:
            st.info("Please upload an image or select from dataset")
            return
        
        st.subheader("🎨 Step 2: Choose Decoder")
        selected_decoder = st.selectbox("Select decoder:", list(BINARY_DECODERS.keys()))
        
        if st.button("🚀 Reconstruct & Analyze", type="primary", use_container_width=True):
            with st.spinner("Processing..."):
                image_tensor = preprocess_image(image)
                
                with torch.no_grad():
                    features = models_dict['encoder'](image_tensor)
                    reconstruction = models_dict['decoders'][selected_decoder](features)
                    
                    pred_orig = models_dict['classifier'](image_tensor)
                    pred_recon = models_dict['classifier'](reconstruction)
                
                st.markdown("---")
                st.markdown("<h2 class='section-header'>Reconstruction Results</h2>", unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**Original Image**")
                    st.image(image, use_column_width=True)
                    
                    orig_pred_idx = pred_orig.argmax().item()
                    orig_class = "Cat" if orig_pred_idx == 0 else "Dog"
                    orig_conf = pred_orig.softmax(dim=1).max().item()
                    st.markdown(f"**Class**: {orig_class}  \n**Confidence**: {orig_conf:.1%}")
                
                with col2:
                    st.markdown(f"**Reconstructed ({selected_decoder})**")
                    recon_image = tensor_to_image(reconstruction)
                    st.image(recon_image, use_column_width=True)
                    
                    recon_pred_idx = pred_recon.argmax().item()
                    recon_class = "Cat" if recon_pred_idx == 0 else "Dog"
                    recon_conf = pred_recon.softmax(dim=1).max().item()
                    st.markdown(f"**Class**: {recon_class}  \n**Confidence**: {recon_conf:.1%}")
                
                with col3:
                    st.markdown("**Decoder Specs**")
                    decoder_info = BINARY_DECODERS[selected_decoder]
                    
                    st.metric("Architecture", decoder_info['architecture'])
                    st.metric("Activation", decoder_info['activation'].title())
                    st.metric("α Parameter", f"{decoder_info['alpha']:.2f}")
                    st.metric("Compaction", f"{decoder_info['compaction_pct']:+.2f}%")
                
                st.markdown("---")
                st.success("✅ Reconstruction complete!")

# ═══════════════════════════════════════════════════════════════════════════════
# 📖 PAGE: ABOUT
# ═══════════════════════════════════════════════════════════════════════════════

def page_about():
    """Project information"""
    st.title("📖 About This Project")
    
    st.markdown("""
    ## Task-Aware Image Reconstruction via Normalized Loss Balancing
    
    **Authors**: Rishabh Ranjan, Mumtaz Parveen, Mithun Raj, Mallanagowda Hosamani  
    **Institution**: ACS College of Engineering, Bangalore  
    **Affiliation**: Visvesvaraya Technological University, Belagavi  
    **Year**: 2025-2026
    
    ---
    
    ### 🎯 Core Innovation
    
    The **Normalized Loss Balancing Framework** solves multi-objective optimization by normalizing via exponential moving averages.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Binary Experiments")
        st.markdown(f"""
        - **Total**: {BINARY_METRICS['total_experiments']} experiments
        - **Architectures**: {BINARY_METRICS['architectures']}
        - **GPU Time**: {BINARY_METRICS['total_gpu_hours']} hours
        - **Baseline Accuracy**: {BINARY_CLASSIFIER['accuracy']:.1%}
        """)
    
    with col2:
        st.subheader("📊 Multiclass Experiments")
        st.markdown(f"""
        - **Total**: {MULTICLASS_METRICS['total_experiments']} experiments
        - **Architectures**: {MULTICLASS_METRICS['architectures']}
        - **GPU Time**: {MULTICLASS_METRICS['total_gpu_hours']} hours
        - **Baseline Accuracy**: {MULTICLASS_CLASSIFIER['accuracy']:.1%}
        """)
    
    st.markdown("---")
    st.subheader("🔐 Privacy Analysis")
    
    specificity = EXTRACTOR_SPECIFICITY
    spec_data = {
        "Training": ["Original", "Reconstructed"],
        "Tested on Original": [
            f"{specificity['trained_on_original']['tested_on_original']:.1%}",
            f"{specificity['trained_on_reconstructions']['tested_on_original']:.1%}"
        ],
        "Tested on Reconstruction": [
            f"{specificity['trained_on_original']['tested_on_reconstructions']:.1%}",
            f"{specificity['trained_on_reconstructions']['tested_on_reconstructions']:.1%}"
        ]
    }
    
    st.table(pd.DataFrame(spec_data))

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Main application"""
    st.sidebar.title("🗂️ Navigation")
    
    page = st.sidebar.radio(
        "Select Page:",
        ["🏠 Home", "📈 Results (Multiclass)", "🔬 Interactive Demo (Binary)", "📖 About"]
    )
    
    st.sidebar.markdown("---")
    
    
    if page == "🏠 Home":
        page_home()
    elif page == "📈 Results (Multiclass)":
        page_results()
    elif page == "🔬 Interactive Demo (Binary)":
        page_demo()
    elif page == "📖 About":
        page_about()

if __name__ == "__main__":
    main()
