# Predenergy 安装指南 (Fixed Version)

本指南将帮助您正确安装和配置 Predenergy 修复版本。

## 📋 系统要求

### 基础要求
- **Python**: 3.8 或更高版本
- **操作系统**: Windows 10+, macOS 10.14+, Linux (Ubuntu 18.04+)
- **内存**: 最少 8GB RAM (推荐 16GB+)
- **存储**: 最少 5GB 可用空间

### 可选要求
- **GPU**: NVIDIA GPU with CUDA 11.0+ (用于加速训练)
- **Docker**: 用于容器化部署

## 🚀 快速安装

### 方法 1: 使用 pip (推荐)

```bash
# 1. 克隆仓库
git clone https://github.com/your-org/predenergy.git
cd predenergy

# 2. 创建虚拟环境
python -m venv predenergy_env

# 3. 激活虚拟环境
# Windows:
predenergy_env\Scripts\activate
# macOS/Linux:
source predenergy_env/bin/activate

# 4. 升级 pip
python -m pip install --upgrade pip

# 5. 安装依赖
pip install -r requirements.txt

# 6. 验证安装
python quick_test.py
```

### 方法 2: 使用 conda

```bash
# 1. 克隆仓库
git clone https://github.com/your-org/predenergy.git
cd predenergy

# 2. 创建 conda 环境
conda create -n predenergy python=3.9
conda activate predenergy

# 3. 安装 PyTorch (根据您的系统选择)
# CPU 版本:
conda install pytorch torchvision torchaudio cpuonly -c pytorch
# GPU 版本 (CUDA 11.8):
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 4. 安装其他依赖
pip install -r requirements.txt

# 5. 验证安装
python quick_test.py
```

## 🔧 详细安装步骤

### 步骤 1: 准备环境

#### 检查 Python 版本
```bash
python --version
# 应显示 Python 3.8.x 或更高
```

#### 安装 Git (如果未安装)
- **Windows**: 从 [git-scm.com](https://git-scm.com/) 下载
- **macOS**: `brew install git` 或从 App Store 安装 Xcode
- **Linux**: `sudo apt-get install git` (Ubuntu/Debian)

### 步骤 2: 获取代码

```bash
# 克隆仓库
git clone https://github.com/your-org/predenergy.git
cd predenergy

# 检查文件结构
ls -la
# 应该看到 models/, scripts/, src/, configs/ 等文件夹
```

### 步骤 3: 设置虚拟环境

#### 使用 venv (Python 内置)
```bash
# 创建虚拟环境
python -m venv predenergy_env

# 激活环境
# Windows Command Prompt:
predenergy_env\Scripts\activate.bat
# Windows PowerShell:
predenergy_env\Scripts\Activate.ps1
# macOS/Linux:
source predenergy_env/bin/activate

# 验证激活
which python  # 应指向虚拟环境
```

#### 使用 virtualenv
```bash
# 安装 virtualenv (如果未安装)
pip install virtualenv

# 创建虚拟环境
virtualenv predenergy_env

# 激活环境
# Windows:
predenergy_env\Scripts\activate
# macOS/Linux:
source predenergy_env/bin/activate
```

### 步骤 4: 安装依赖

#### 基础安装
```bash
# 升级 pip
python -m pip install --upgrade pip

# 安装核心依赖
pip install -r requirements.txt
```

#### GPU 支持 (可选)
如果您有 NVIDIA GPU 并希望使用 CUDA 加速：

```bash
# 检查 CUDA 版本
nvidia-smi

# 安装对应的 PyTorch 版本
# 访问 https://pytorch.org/get-started/locally/ 获取具体命令
# 例如，对于 CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 开发依赖 (可选)
```bash
# 安装开发和测试依赖
pip install pytest pytest-cov black flake8 jupyter
```

### 步骤 5: 验证安装

#### 运行快速测试
```bash
python quick_test.py
```

期望输出：
```
🚀 Predenergy Fixed Implementation - Quick Test Suite
============================================================
🔍 Testing imports...
  ✅ Unified config import successful
  ✅ Fixed model import successful
  ✅ Main model class import successful
  ✅ Config loader import successful

🔧 Testing configuration system...
  ✅ Basic configuration creation successful
  ✅ Configuration validation successful
  ✅ Configuration serialization successful
...
📊 Overall: 6/6 tests passed (100.0%)

🎉 All tests passed! The fixed implementation is working correctly.
```

#### 测试训练脚本
```bash
# 查看帮助信息
python scripts/fixed_train_predenergy.py --help
```

#### 测试 API 服务器
```bash
# 启动服务器 (在另一个终端)
python src/fixed_api_predenergy.py

# 在原终端测试
curl http://localhost:8000/health
```

## 🐛 常见问题解决

### 问题 1: 导入错误
```
ImportError: No module named 'models.Predenergy'
```
**解决方案:**
```bash
# 确保在项目根目录
pwd  # 应显示 .../predenergy

# 检查 PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### 问题 2: PyTorch 版本冲突
```
RuntimeError: The NVIDIA driver on your system is too old
```
**解决方案:**
```bash
# 安装 CPU 版本的 PyTorch
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 问题 3: 内存不足
```
RuntimeError: CUDA out of memory
```
**解决方案:**
```python
# 在配置中减少批次大小
config = PredenergyUnifiedConfig(
    batch_size=8,  # 从 32 减少到 8
    d_model=256,   # 从 512 减少到 256
)
```

### 问题 4: 权限错误 (Windows)
```
PermissionError: [WinError 5] Access is denied
```
**解决方案:**
```bash
# 以管理员身份运行命令提示符
# 或使用用户安装模式
pip install --user -r requirements.txt
```

### 问题 5: SSL 证书错误
```
SSL: CERTIFICATE_VERIFY_FAILED
```
**解决方案:**
```bash
# 升级证书
pip install --upgrade certifi

# 或使用信任的主机
pip install -r requirements.txt --trusted-host pypi.org --trusted-host pypi.python.org
```

## 🔄 环境管理

### 激活/停用环境
```bash
# 激活环境
source predenergy_env/bin/activate  # macOS/Linux
predenergy_env\Scripts\activate     # Windows

# 停用环境
deactivate
```

### 更新依赖
```bash
# 激活环境后
pip install --upgrade -r requirements.txt
```

### 导出环境配置
```bash
# 导出当前环境
pip freeze > environment.txt

# 在其他机器上重建环境
pip install -r environment.txt
```

### 清理环境
```bash
# 停用并删除虚拟环境
deactivate
rm -rf predenergy_env  # macOS/Linux
rmdir /s predenergy_env  # Windows
```

## 🐳 Docker 安装 (可选)

### 创建 Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 复制代码
COPY . .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["python", "src/fixed_api_predenergy.py", "--host", "0.0.0.0"]
```

### 构建和运行容器
```bash
# 构建镜像
docker build -t predenergy:fixed .

# 运行容器
docker run -p 8000:8000 predenergy:fixed

# 运行测试
docker run predenergy:fixed python quick_test.py
```

## 📚 下一步

安装完成后，您可以：

1. **阅读文档**: 查看 `README.md` 了解详细使用方法
2. **运行示例**: 尝试 `scripts/fixed_train_predenergy.py`
3. **使用 API**: 启动 `src/fixed_api_predenergy.py`
4. **查看修复**: 阅读 `FIXES_SUMMARY.md` 了解改进内容

## 🆘 获取帮助

如果遇到问题：

1. **检查日志**: 查看详细的错误信息
2. **运行测试**: 使用 `python quick_test.py` 诊断问题
3. **查看文档**: README.md 和 FIXES_SUMMARY.md
4. **提交 Issue**: 在 GitHub 上报告问题
5. **社区讨论**: 加入技术讨论群

---

**安装成功标志**: `python quick_test.py` 显示所有测试通过 ✅