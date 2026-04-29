#!/usr/bin/env bash
# ============================================
# 基于用户数据特征画像的终端数据泄露检测方法
# 一键部署与管理脚本
# ============================================
#
# 用法:
#   ./deploy.sh setup       # 首次环境初始化
#   ./deploy.sh start       # 启动检测服务
#   ./deploy.sh demo        # 运行完整演示
#   ./deploy.sh experiment  # 运行全部实验
#   ./deploy.sh stop        # 停止服务
#   ./deploy.sh status      # 查看服务状态
#   ./deploy.sh logs        # 查看日志
#   ./deploy.sh clean       # 清理容器和卷

set -euo pipefail

# ─── 颜色输出 ───
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ─── 项目根目录 ───
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG_FILE="${DLP_CONFIG_PATH:-configs/default_config.yaml}"

# ============================================
# 环境检查
# ============================================
check_prerequisites() {
    info "检查运行环境..."

    # Python
    if command -v python3 &>/dev/null; then
        PY_VER=$(python3 --version 2>&1 | cut -d' ' -f2)
        ok "Python $PY_VER"
    else
        err "未找到 Python3，请安装 Python >= 3.10"
    fi

    # PyTorch + CUDA
    if python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}')" 2>/dev/null; then
        ok "PyTorch 已安装"
    else
        warn "PyTorch 未安装或不可用（将使用 CPU 模式）"
    fi

    # Docker (可选)
    if command -v docker &>/dev/null; then
        DOCKER_VER=$(docker --version | cut -d' ' -f3 | tr -d ',')
        ok "Docker $DOCKER_VER"
    else
        warn "Docker 未安装（跳过容器化部署）"
    fi

    # NVIDIA GPU
    if command -v nvidia-smi &>/dev/null; then
        GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)
        ok "GPU: $GPU_INFO"
    else
        warn "未检测到 NVIDIA GPU（将使用 CPU 模式）"
    fi
}

# ============================================
# 首次初始化
# ============================================
cmd_setup() {
    info "============================================"
    info "  首次环境初始化"
    info "============================================"

    check_prerequisites

    # 创建目录结构
    info "创建目录结构..."
    mkdir -p data/raw/rvlcdip data/raw/enron data/raw/cert_r4.2
    mkdir -p data/processed
    mkdir -p models
    mkdir -p experiments/logs experiments/figures experiments/results
    mkdir -p logs
    ok "目录结构已创建"

    # 安装 Python 依赖
    info "安装 Python 依赖..."
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt 2>&1 | tail -5
        ok "依赖安装完成"
    else
        err "requirements.txt 不存在"
    fi

    # 下载模型权重
    info "下载模型权重..."
    if [ -f "scripts/download_models.py" ]; then
        python3 scripts/download_models.py --output models/ || warn "模型下载失败（可手动下载）"
    fi

    # 准备数据集
    info "准备数据集..."
    if [ -f "scripts/prepare_datasets.py" ]; then
        python3 scripts/prepare_datasets.py --output data/ || warn "数据集准备失败（可手动准备）"
    fi

    # 验证安装
    info "验证安装..."
    python3 -c "
import sys
modules = [
    'torch', 'transformers', 'sentence_transformers',
    'numpy', 'scipy', 'sklearn', 'yaml', 'faiss',
]
missing = []
for m in modules:
    try:
        __import__(m)
    except ImportError:
        missing.append(m)
if missing:
    print(f'缺少模块: {missing}', file=sys.stderr)
    sys.exit(1)
print('所有核心模块已就绪')
" && ok "模块验证通过" || warn "部分模块缺失（请检查 requirements.txt）"

    echo ""
    ok "============================================"
    ok "  初始化完成！"
    ok "  运行 './deploy.sh demo' 查看完整演示"
    ok "============================================"
}

# ============================================
# 启动服务
# ============================================
cmd_start() {
    info "启动检测服务..."

    if command -v docker &>/dev/null && [ -f "docker-compose.yml" ]; then
        info "使用 Docker Compose 启动..."
        docker compose up -d dlp-server
        ok "服务已启动 (容器模式)"
        info "API 端口: http://localhost:8080"
        info "gRPC 端口: localhost:50051"
    else
        info "使用本地 Python 启动..."
        nohup python3 main.py serve \
            --watch "${1:-data/}" \
            --config "$CONFIG_FILE" \
            > logs/server.log 2>&1 &
        echo $! > logs/server.pid
        ok "检测服务已启动 (PID: $(cat logs/server.pid))"
        info "日志: tail -f logs/server.log"
    fi
}

# ============================================
# 运行演示
# ============================================
cmd_demo() {
    info "运行完整流程演示..."
    echo ""

    if command -v docker &>/dev/null && [ -f "docker-compose.yml" ]; then
        docker compose run --rm dlp-server \
            python main.py demo --config /app/configs/default_config.yaml
    else
        python3 main.py demo --config "$CONFIG_FILE"
    fi
}

# ============================================
# 运行实验
# ============================================
cmd_experiment() {
    info "运行全部实验..."
    info "预计耗时: GPU 模式 ~4h, CPU 模式 ~24h"
    echo ""

    if command -v docker &>/dev/null && [ -f "docker-compose.yml" ]; then
        docker compose --profile experiment up experiment-runner
    else
        python3 scripts/run_all_experiments.py \
            --config "$CONFIG_FILE" \
            --output experiments/results
    fi

    ok "实验完成，结果保存在 experiments/results/"
}

# ============================================
# 停止服务
# ============================================
cmd_stop() {
    info "停止检测服务..."

    if command -v docker &>/dev/null; then
        docker compose down 2>/dev/null || true
    fi

    for pidfile in logs/server.pid logs/monitor.pid; do
        if [ -f "$pidfile" ]; then
            PID=$(cat "$pidfile")
            if kill -0 "$PID" 2>/dev/null; then
                kill "$PID"
                rm -f "$pidfile"
                ok "本地进程已停止 (PID: $PID)"
            fi
        fi
    done

    ok "服务已停止"
}

# ============================================
# 查看状态
# ============================================
cmd_status() {
    info "服务状态:"
    echo ""

    # Docker 容器
    if command -v docker &>/dev/null; then
        CONTAINERS=$(docker compose ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}" 2>/dev/null || echo "")
        if [ -n "$CONTAINERS" ]; then
            echo "$CONTAINERS"
        else
            info "无运行中的 Docker 容器"
        fi
    fi

    # 本地进程
    for pidfile in logs/server.pid logs/monitor.pid; do
        if [ -f "$pidfile" ]; then
            PID=$(cat "$pidfile")
            name=$(basename "$pidfile" .pid)
            if kill -0 "$PID" 2>/dev/null; then
                ok "本地 ${name} 进程运行中 (PID: $PID)"
            else
                warn "本地 ${name} 进程已退出 (PID: $PID)"
            fi
        fi
    done

    # GPU 状态
    echo ""
    if command -v nvidia-smi &>/dev/null; then
        nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total \
            --format=csv,noheader 2>/dev/null || true
    fi

    # 实验结果
    echo ""
    RESULT_COUNT=$(find experiments/results -name "*.csv" 2>/dev/null | wc -l)
    info "实验结果文件: ${RESULT_COUNT} 个 CSV"
}

# ============================================
# 查看日志
# ============================================
cmd_logs() {
    if command -v docker &>/dev/null; then
        docker compose logs -f --tail=100 dlp-server 2>/dev/null || true
    fi

    if [ -f "logs/server.log" ]; then
        info "本地日志 (最近 50 行):"
        tail -50 logs/server.log
    elif [ -f "logs/monitor.log" ]; then
        info "本地日志 (最近 50 行):"
        tail -50 logs/monitor.log
    fi
}

# ============================================
# 扫描单个文件
# ============================================
cmd_scan() {
    if [ -z "${1:-}" ]; then
        err "用法: ./deploy.sh scan <文件路径> [--user <用户ID>]"
    fi

    python3 main.py scan "$@" --config "$CONFIG_FILE"
}

# ============================================
# 生成报告
# ============================================
cmd_report() {
    python3 main.py report
}

# ============================================
# 清理
# ============================================
cmd_clean() {
    warn "即将清理所有容器、镜像和数据卷..."
    read -p "确认继续？[y/N] " -r
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker compose down -v --rmi local 2>/dev/null || true
        rm -rf logs/*.log logs/*.pid
        rm -rf __pycache__ src/__pycache__
        ok "清理完成"
    else
        info "已取消"
    fi
}

# ============================================
# 主入口
# ============================================
case "${1:-help}" in
    setup)      cmd_setup ;;
    start)      cmd_start "${2:-}" ;;
    demo)       cmd_demo ;;
    experiment) cmd_experiment ;;
    stop)       cmd_stop ;;
    status)     cmd_status ;;
    logs)       cmd_logs ;;
    scan)       shift; cmd_scan "$@" ;;
    report)     cmd_report ;;
    clean)      cmd_clean ;;
    help|*)
        echo ""
        echo "  基于用户数据特征画像的终端数据泄露检测系统"
        echo "  Terminal Data Leakage Detection System"
        echo ""
        echo "  用法: ./deploy.sh <命令>"
        echo ""
        echo "  命令:"
        echo "    setup       首次环境初始化（安装依赖、下载模型）"
        echo "    start       启动检测服务（Docker 或本地模式）"
        echo "    demo        运行完整流程演示"
        echo "    experiment  运行全部实验（~4h GPU / ~24h CPU）"
        echo "    scan <file> 扫描单个文件"
        echo "    report      生成实验报告"
        echo "    stop        停止服务"
        echo "    status      查看运行状态"
        echo "    logs        查看日志"
        echo "    clean       清理容器和数据"
        echo ""
        echo "  示例:"
        echo "    ./deploy.sh setup && ./deploy.sh demo"
        echo "    ./deploy.sh scan report.pdf --user alice"
        echo "    ./deploy.sh start /data/watch"
        echo ""
        ;;
esac
