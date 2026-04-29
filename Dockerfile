# ============================================
# 基于用户数据特征画像的终端数据泄露检测方法
# Docker 部署配置
# ============================================

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

LABEL maintainer="HUST Cyberspace Security Lab"
LABEL description="Terminal Data Leakage Detection Based on User Data Feature Profiling"

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# 设置 Python 环境
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# 设置工作目录
WORKDIR /app

# 复制依赖文件并安装
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY configs/ ./configs/
COPY main.py setup.py requirements.txt ./

# 安装项目为可编辑包
RUN pip install --no-cache-dir -e .

# 创建数据和模型挂载点
VOLUME ["/app/data", "/app/models", "/app/experiments", "/app/logs"]

# 暴露端口
EXPOSE 8080 50051 6006

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import src; print('OK')" || exit 1

# 默认入口：启动检测系统
ENTRYPOINT ["python", "main.py"]
CMD ["serve", "--config", "configs/default_config.yaml"]
