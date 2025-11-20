# Affective Subtitler - 环境设置指南

## 快速开始

1. **激活环境**：
```bash
conda activate affective_subtitler
```

2. **安装依赖**：
```bash
pip install -r requirements.txt --default-timeout=1000
```

3. **运行应用**：
```bash
./run_app.sh
```

## 环境说明

- **Python**: 3.10
- **主要依赖**: WhisperX, Transformers, Streamlit
- **注意**: WhisperX 使用 CPU 模式以避免 cuDNN 兼容性问题（SER 仍用 GPU）

## 已知的版本警告

你可能会看到以下警告（**可以忽略**）：
- `pyannote.audio 0.0.1 vs 3.4.0` 
- `torch 1.10.0 vs 2.8.0`

这些警告不影响功能，只是提示模型训练和推理环境的版本差异。
