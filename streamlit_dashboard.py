"""
╔════════════════════════════════════════════════════════════════════════════════╗
║ TASK-AWARE IMAGE RECONSTRUCTION - DEMO (6 VARIANTS, 5 SAMPLES)               ║
║ Original + Deep AE + SimpleTanh + SimpleSig + MRTanh + MRSig side by side    ║
╚════════════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
import matplotlib.pyplot as plt
from torchvision import models, transforms, datasets
from torchvision.models import resnet18, ResNet18_Weights
import glob
import io

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ═════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Task-Aware Reconstruction",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    :root {
        --primary: #667eea;
        --primary-dark: #764ba2;
        --accent: #00d4ff;
        --bg-dark: #0a0e27;
        --bg-card: #1a1f3a;
        --border: #2d3561;
        --text-primary: #e8eaf6;
        --text-secondary: #b0bec5;
    }
    
    .stApp {
        background-color: var(--bg-dark);
        color: var(--text-primary);
    }
    
    .header-premium {
        font-size: 2.4em;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 8px;
        letter-spacing: -0.5px;
    }
    
    .subheader-premium {
        font-size: 1.1em;
        color: var(--text-secondary);
        margin-bottom: 24px;
        font-weight: 300;
        letter-spacing: 0.5px;
    }
    
    .section-title {
        font-size: 1.3em;
        font-weight: 600;
        color: var(--text-primary);
        margin-top: 20px;
        margin-bottom: 12px;
        border-bottom: 1px solid var(--border);
        padding-bottom: 8px;
    }
    
    .metric-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%);
        border: 1px solid var(--border);
        color: var(--text-primary);
        padding: 16px;
        border-radius: 12px;
        text-align: center;
        font-weight: 500;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        border-color: #667eea;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.25) 0%, rgba(118, 75, 162, 0.25) 100%);
    }
    
    .metric-value {
        font-size: 2.2em;
        margin: 8px 0;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700;
    }
    
    .metric-label {
        font-size: 0.8em;
        color: var(--text-secondary);
        margin-top: 4px;
    }
    
    .text-muted {
        color: var(--text-secondary);
        font-size: 0.9em;
        line-height: 1.6;
    }
    
    .text-accent {
        color: #00d4ff;
        font-weight: 500;
    }
    
    .info-box {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.1) 0%, rgba(102, 126, 234, 0.1) 100%);
        border-left: 3px solid #00d4ff;
        padding: 12px 16px;
        border-radius: 6px;
        margin: 12px 0;
        font-size: 0.9em;
        color: var(--text-primary);
    }
    
    .divider-premium {
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--border), transparent);
        margin: 24px 0;
    }
    
    .experimental-badge {
        display: inline-block;
        background: linear-gradient(135deg, rgba(255, 107, 107, 0.2) 0%, rgba(255, 159, 107, 0.2) 100%);
        border: 1px solid rgba(255, 107, 107, 0.5);
        color: #ff6b6b;
        padding: 4px 12px;
        border-radius: 6px;
        font-size: 0.75em;
        font-weight: 600;
        margin-left: 8px;
    }
    
    [data-testid="stDataFrame"] {
        background-color: var(--bg-card);
    }
    
    [data-testid="stSidebar"] {
        background-color: var(--bg-dark);
        border-right: 1px solid var(--border);
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        font-weight: 600;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        opacity: 0.9;
        box-shadow: 0 8px 16px rgba(102, 126, 234, 0.3);
    }
    
    </style>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# 🏗️ MODELS
# ═════════════════════════════════════════════════════════════════════════════

class ResNet18Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.features = nn.Sequential(
            base_model.conv1, base_model.bn1, base_model.relu, base_model.maxpool,
            base_model.layer1, base_model.layer2, base_model.layer3
        )
    
    def forward(self, x):
        return self.features(x)

class SimpleDecoder(nn.Module):
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

class DeepAutoencoder_Encoder(nn.Module):
    def __init__(self, z_channels=512):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, z_channels, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.enc(x)

class DeepAutoencoder_Decoder(nn.Module):
    def __init__(self, z_channels=512):
        super().__init__()
        self.dec = nn.Sequential(
            nn.Conv2d(z_channels, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 128, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 3, 1, 1),
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.dec(z)

class DeepAutoencoder(nn.Module):
    def __init__(self, z_channels=512):
        super().__init__()
        self.encoder = DeepAutoencoder_Encoder(z_channels)
        self.decoder = DeepAutoencoder_Decoder(z_channels)
    
    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon

class MulticlassClassifier(nn.Module):
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

# ═════════════════════════════════════════════════════════════════════════════
# 💾 MODEL LOADING
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_models():
    models_dict = {}
    
    with st.spinner("⚡ Loading models..."):
        encoder = ResNet18Encoder()
        encoder.eval()
        models_dict['encoder'] = encoder
        
        classifier = MulticlassClassifier()
        classifier.eval()
        models_dict['classifier'] = classifier
        
        # Binary classifier - load ResNet18 directly (not wrapped)
        binary_classifier = resnet18(weights=ResNet18_Weights.DEFAULT)
        num_ftrs = binary_classifier.fc.in_features
        binary_classifier.fc = nn.Linear(num_ftrs, 2)
        binary_classifier.eval()
        try:
            state_dict = torch.load('model.pth', map_location='cpu', weights_only=True)
            binary_classifier.load_state_dict(state_dict)
        except Exception as e:
            st.warning(f"⚠️ Binary classifier load failed: {str(e)}")
        models_dict['binary_classifier'] = binary_classifier
        
        deep_ae = DeepAutoencoder(z_channels=512)
        deep_ae.eval()
        try:
            state_dict = torch.load('MulticlassExpansion/deep_autoencoder.pth', map_location='cpu', weights_only=True)
            deep_ae.load_state_dict(state_dict)
        except Exception as e:
            st.warning(f"⚠️ Deep AE load failed: {str(e)}")
        models_dict['deep_ae'] = deep_ae
        
        decoders = {}
        decoder_configs = [
            ('SimpleTanh', 'simple', 'tanh', 'MulticlassExpansion/SSigmoid_TargetAlpha/decoder*.pth'),
            ('SimpleSigmoid', 'simple', 'sigmoid', 'MulticlassExpansion/SSigmoid_TargetAlpha/decoder*.pth'),
            ('MRTanh', 'mirrored', 'tanh', 'MulticlassExpansion/MRTanh_TargetAlpha/decoder*.pth'),
            ('MRSigmoid', 'mirrored', 'sigmoid', 'MulticlassExpansion/MRSigmoid_TargetAlpha/decoder*.pth'),
        ]
        
        for name, arch, act, pattern in decoder_configs:
            try:
                decoder_class = SimpleDecoder if arch == 'simple' else MirroredResNetDecoder
                decoder = decoder_class(activation=act)
                decoder.eval()
                
                pth_files = glob.glob(pattern)
                if pth_files:
                    state_dict = torch.load(pth_files[0], map_location='cpu', weights_only=True)
                    decoder.load_state_dict(state_dict)
                
                decoders[name] = decoder
            except Exception as e:
                st.warning(f"⚠️ Could not load {name}: {str(e)}")
        
        models_dict['decoders'] = decoders
    
    return models_dict

@st.cache_resource
def load_cifar10():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                           std=[0.247, 0.243, 0.261])
    ])
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    return test_dataset

# ═════════════════════════════════════════════════════════════════════════════
# 🖼️ UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

def denormalize_tensor(tensor):
    """Convert normalized tensor to PIL image"""
    img = tensor.clone().squeeze(0)
    for c in range(3):
        img[c] = img[c] * torch.tensor([0.247, 0.243, 0.261])[c] + torch.tensor([0.4914, 0.4822, 0.4465])[c]
    img = torch.clamp(img, 0, 1)
    return transforms.ToPILImage()(img)

def prepare_image(pil_img):
    """Prepare PIL image for model input"""
    pil_img = pil_img.resize((32, 32), Image.LANCZOS)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                           std=[0.247, 0.243, 0.261])
    ])
    return transform(pil_img).unsqueeze(0)

# ═════════════════════════════════════════════════════════════════════════════
# 📄 PAGE 1: INTRO
# ═════════════════════════════════════════════════════════════════════════════

def page_intro():
    st.markdown('<div class="header-premium">🖼️ Task-Aware Image Reconstruction</div>', unsafe_allow_html=True)
    st.markdown('<div class="subheader-premium">Normalized Loss Balancing for Edge Vision</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="divider-premium"></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown('<div class="section-title">⚡ The Problem</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="text-muted">
        Traditional image compression (JPEG, WebP) optimizes for <span class="text-accent">human perception</span>.
        <br><br>
        But on edge devices, this creates a critical mismatch:
        <br>• Wastes bandwidth transmitting task-irrelevant details
        <br>• Destroys features classifiers depend on
        <br>• Machines and humans need different representations
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="section-title">💡 Our Solution</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="text-muted">
        <strong>Task-aware compression</strong> via normalized loss balancing.
        <br><br>
        Simultaneously optimize two objectives:
        <br>• Classification accuracy (task performance)
        <br>• Reconstruction quality (compression efficiency)
        <br><br>
        Controlled by interpretable <span class="text-accent">α ∈ [0, 1]</span> parameter.
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="divider-premium"></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-title">📊 Results at a Glance</div>', unsafe_allow_html=True)
    
    metric_cols = st.columns(4, gap="small")
    metrics = [
        ("Accuracy", "78.40%", "vs 77.71% baseline"),
        ("Compaction", "18% avg", "22–23% for simple decoders"),
        ("Privacy", "11.23%", "cross-model extractor specificity"),
        ("Experiments", "464", "on CIFAR-10"),
    ]
    
    for col, (label, value, desc) in zip(metric_cols, metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
            <div class="metric-label" style="font-size: 0.7em; margin-top: 6px; color: #00d4ff;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('<div class="divider-premium"></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-title">🔍 Why This Matters</div>', unsafe_allow_html=True)
    
    comp_data = {
        "Method": ["JPEG (traditional)", "Deep Autoencoder", "Our Method"],
        "Accuracy": ["~45%", "69.37%", "78.40%"],
        "Focus": ["Human perception", "Reconstruction fidelity", "Task performance"],
        "Status": ["❌ Task-agnostic", "⚠️ Incomplete", "✓ Balanced"],
    }
    
    comp_df = pd.DataFrame(comp_data)
    st.dataframe(comp_df, use_container_width=True, hide_index=True)
    
    st.markdown("""
    <div class="info-box">
    <strong>Key insight:</strong> Deep autoencoders optimize MSE (pixel-perfect reconstruction) which actually <em>harms</em> 
    classification accuracy. We balance both for optimal edge deployment.
    </div>
    """, unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# 🎨 PAGE 2: DEMO - 6 VARIANTS, 5 SAMPLES
# ═════════════════════════════════════════════════════════════════════════════

def page_demo():
    st.markdown('<div class="header-premium">🎨 Reconstruction Demo</div>', unsafe_allow_html=True)
    st.markdown('<div class="subheader-premium">6 variants side by side on CIFAR-10 test images</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="divider-premium"></div>', unsafe_allow_html=True)
    
    dataset = load_cifar10()
    models = load_models()
    encoder = models['encoder']
    deep_ae = models['deep_ae']
    decoders = models['decoders']
    
    cifar_classes = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
    
    tab_dataset, tab_upload = st.tabs(["📊 Dataset", "⬆️ Upload"])
    
    with tab_dataset:
        st.markdown('<div class="section-title">📌 Select Class</div>', unsafe_allow_html=True)
        selected_class = st.selectbox("", cifar_classes, key="demo_class", label_visibility="collapsed")
        
        st.markdown('<div class="divider-premium"></div>', unsafe_allow_html=True)
        
        class_idx = cifar_classes.index(selected_class)
        class_indices = [i for i, (_, y) in enumerate(dataset) if y == class_idx][:5]
        
        if not class_indices:
            st.error(f"No {selected_class} samples in dataset")
            return
        
        st.markdown(f'<div class="section-title">{selected_class} — 5 Samples</div>', unsafe_allow_html=True)
        
        variants = [
            ('Original', None, None, 'Input'),
            ('Deep AE', 'deep_ae', None, 'Deep Autoencoder Baseline'),
            ('SimpleTanh', 'SimpleTanh', 'decoders', 'Simple Decoder (Tanh)'),
            ('SimpleSig', 'SimpleSigmoid', 'decoders', 'Simple Decoder (Sigmoid)'),
            ('MRTanh', 'MRTanh', 'decoders', 'Mirrored ResNet (Tanh)'),
            ('MRSig', 'MRSigmoid', 'decoders', 'Mirrored ResNet (Sigmoid)'),
        ]
        
        fig, axes = plt.subplots(5, 6, figsize=(16, 13.5))
        fig.patch.set_facecolor('#0a0e27')
        
        with st.spinner("🔄 Rendering reconstructions..."):
            for row_idx, img_idx in enumerate(class_indices):
                img_tensor, label = dataset[img_idx]
                img_batch = img_tensor.unsqueeze(0)
                
                with torch.no_grad():
                    latent = encoder(img_batch)
                
                for col_idx, (short_name, model_key, model_type, full_label) in enumerate(variants):
                    ax = axes[row_idx, col_idx]
                    
                    try:
                        if short_name == 'Original':
                            img_pil = denormalize_tensor(img_tensor)
                        elif short_name == 'Deep AE':
                            with torch.no_grad():
                                recon = deep_ae(img_batch)
                            img_pil = denormalize_tensor(recon)
                        else:
                            if model_type == 'decoders' and model_key in decoders:
                                with torch.no_grad():
                                    recon = decoders[model_key](latent)
                                img_pil = denormalize_tensor(recon)
                            else:
                                ax.text(0.5, 0.5, 'N/A', ha='center', va='center', color='#b0bec5', fontsize=11, fontweight='bold')
                                ax.axis('off')
                                continue
                        
                        ax.imshow(img_pil)
                        
                        if row_idx == 0:
                            ax.set_title(full_label, fontsize=9.5, color='#00d4ff', fontweight='bold', pad=8)
                        
                        if col_idx == 0:
                            ax.text(-0.35, 0.5, f'Sample {row_idx + 1}', 
                                   transform=ax.transAxes, fontsize=9, color='#667eea',
                                   fontweight='bold', va='center', ha='right')
                        
                        for spine in ax.spines.values():
                            spine.set_edgecolor('#2d3561')
                            spine.set_linewidth(0.5)
                        
                        ax.axis('off')
                    
                    except Exception as e:
                        ax.text(0.5, 0.5, 'Error', ha='center', va='center', color='#ff6b6b', fontsize=10)
                        ax.axis('off')
        
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()
        
        st.markdown("""
        <div class="info-box">
        <strong>Reading the grid:</strong> Each row shows a different CIFAR-10 sample. Columns represent different reconstruction methods, from baseline (Deep AE) to our task-aware variants.
        <br><br>
        <strong>Key observation:</strong> Simple decoders achieve aggressive compression while preserving task-critical features. Mirrored ResNet preserves perceptual quality with minimal compaction.
        </div>
        """, unsafe_allow_html=True)
    
    with tab_upload:
        st.markdown('<div class="section-title">📤 Upload Custom Image <span class="experimental-badge">EXPERIMENTAL</span></div>', unsafe_allow_html=True)
        st.markdown('<div class="text-muted">Upload any image (will be resized to 32×32) — Classification: Dog vs Cat</div>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("", type=['jpg', 'jpeg', 'png', 'bmp'], label_visibility="collapsed")
        
        if uploaded_file is not None:
            img = Image.open(uploaded_file).convert('RGB')
            img_tensor = prepare_image(img)
            
            st.markdown('<div class="divider-premium"></div>', unsafe_allow_html=True)
            
            binary_classifier = models['binary_classifier']
            
            with torch.no_grad():
                orig_logits = binary_classifier(img_tensor)
                orig_pred = torch.argmax(orig_logits).item()
            
            binary_classes = ['🐕 Dog', '🐈 Cat']
            
            st.markdown('<div class="section-title">Original Image</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="text-accent">Predicted: <strong>{binary_classes[orig_pred]}</strong></div>', unsafe_allow_html=True)
            
            st.markdown('<div class="divider-premium"></div>', unsafe_allow_html=True)
            
            with torch.no_grad():
                latent = encoder(img_tensor)
            
            variants_upload = [
                ('Deep AE', 'deep_ae', None, 'Deep Autoencoder'),
                ('SimpleTanh', 'SimpleTanh', 'decoders', 'Simple (Tanh)'),
                ('SimpleSig', 'SimpleSigmoid', 'decoders', 'Simple (Sigmoid)'),
                ('MRTanh', 'MRTanh', 'decoders', 'Mirrored (Tanh)'),
                ('MRSig', 'MRSigmoid', 'decoders', 'Mirrored (Sigmoid)'),
            ]
            
            st.markdown('<div class="section-title">🎨 Reconstructions & Classifications</div>', unsafe_allow_html=True)
            
            cols = st.columns(5)
            
            for col_idx, (short_name, model_key, model_type, full_label) in enumerate(variants_upload):
                with cols[col_idx]:
                    try:
                        if short_name == 'Deep AE':
                            with torch.no_grad():
                                recon = deep_ae(img_tensor)
                        else:
                            if model_type == 'decoders' and model_key in decoders:
                                with torch.no_grad():
                                    recon = decoders[model_key](latent)
                            else:
                                st.error("N/A")
                                continue
                        
                        img_pil = denormalize_tensor(recon)
                        
                        with torch.no_grad():
                            recon_logits = binary_classifier(recon)
                            recon_pred = torch.argmax(recon_logits).item()
                        
                        st.image(img_pil, use_container_width=True)
                        st.markdown(f"**{full_label}**", unsafe_allow_html=True)
                        st.markdown(f"<div style='font-size: 0.9em; color: #00d4ff;'>{binary_classes[recon_pred]}</div>", unsafe_allow_html=True)
                    
                    except Exception as e:
                        st.error(str(e)[:50])


# ═════════════════════════════════════════════════════════════════════════════
# 📈 PAGE 3: RESULTS
# ═════════════════════════════════════════════════════════════════════════════

def page_results():
    st.markdown('<div class="header-premium">📈 Results & Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="subheader-premium">Accuracy, compression, and trade-offs</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="divider-premium"></div>', unsafe_allow_html=True)
    
    acc_col, comp_col = st.columns(2, gap="large")
    
    with acc_col:
        st.markdown('<div class="section-title">🎯 Classification Accuracy</div>', unsafe_allow_html=True)
        
        results_data = {
            'Method': ['Baseline', 'Our Best', 'Deep AE', 'JPEG'],
            'Accuracy': [77.71, 78.40, 69.37, 45.0],
            'Δ': ['—', '+0.69%', '-8.34%', '-32.71%']
        }
        df = pd.DataFrame(results_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    
    with comp_col:
        st.markdown('<div class="section-title">📦 Compaction by Variant</div>', unsafe_allow_html=True)
        
        compaction_data = {
            'Decoder': ['SimpleTanh', 'SimpleSigmoid', 'MRTanh', 'MRSigmoid'],
            'Compaction %': [22.43, 23.24, 0.94, -10.92],
        }
        df_comp = pd.DataFrame(compaction_data)
        st.dataframe(df_comp, use_container_width=True, hide_index=True)

# ═════════════════════════════════════════════════════════════════════════════
# 🎛️ MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    pages = {
        "🏠 Intro": page_intro,
        "🎨 Demo": page_demo,
        "📈 Results": page_results,
    }
    
    st.sidebar.markdown("## Navigation")
    page = st.sidebar.radio("", list(pages.keys()), key="nav", label_visibility="collapsed")
    
    pages[page]()

if __name__ == "__main__":
    main()
