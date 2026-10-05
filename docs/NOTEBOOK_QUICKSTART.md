# Notebook 启动

入口：[Launch_Demo.ipynb](../notebooks/Launch_Demo.ipynb)。本地 Jupyter 可打开完整 Web UI；Colab 通过 Notebook 内的本机 HTTP 查询并显示结果，不公开服务，也不创建隧道。

## 获取与更新

使用仓库实际的私有 Git URL；先完成 Git 凭据配置，不要将 token 放进 URL。

```bash
git clone https://github.com/ShellWee/inspection-demo.git
cd inspection-demo
```

以后在该目录运行 `git pull --ff-only`，再打开 `notebooks/Launch_Demo.ipynb`。若有本地修改，先保存它们并处理 Git 报告的冲突。代码、Notebook 与构建好的前端在 Git 中；大型 IFC、模型和索引资产另行取得。

如果在 Colab 只打开了 Notebook，可在第一格填写无凭据的 `GIT_URL`，克隆完整仓库到 `inspection-demo-hf`。私有仓库访问需要当前 Colab 环境已配置 Git 认证；浏览器能够打开私有 Notebook 不代表 Colab 的 Git 已登录。

## 按顺序运行

1. 使用 Python 3.10+ 的 Notebook 内核。应用本身固定使用 Python 3.12，启动器通过 `uv` 创建仓库的 `.venv`，不会要求内核安装 Torch/FastAPI。
2. 配置 `ASSETS_DIR`（默认 `REPO_ROOT / "assets"`）、`DEVICE`（默认 `"cpu"`）和 `PORT`（默认 `7860`）。可以直接指向已有资产包；该目录须包含 `manifest.json` 和清单中的完整文件。
3. 确保内核可调用 `uv`。如需安装，把 `INSTALL_UV=True` 后运行安装格；它安装 `uv==0.12.9`。运行 setup 格后，依赖按 `uv.lock` 安装，不安装开发依赖。首次安装会下载 Python/依赖；不需要 Node.js 编译前端。
4. 验证资产。需要下载时，先在同一运行环境完成 Hugging Face 登录并获准访问私有数据集，再设 `DOWNLOAD_ASSETS=True`。下载锁定数据集版本，默认不会自动下载。HF 访问权限与 Git 访问权限相互独立。
5. 启动并等待就绪。本地打开输出的 `http://127.0.0.1:7860`。Colab 使用查询格查看结果；浏览器中的 localhost 并非 Colab 运行机器，不要创建公开转发。
6. 用完把 `STOP_SERVER=True`，执行停止格。同一个 `.demo-state` 目录支持跨单元格检查/停止；重启内核后重新运行配置与函数定义即可，不必重新安装或下载。

CUDA 需要兼容的 NVIDIA GPU、驱动和锁定的 Torch CUDA 运行时。Linux/Colab 可以选择 `DEVICE="cuda"`；Windows 的当前锁定 Torch 使用 CPU，选择 CPU。启动器验证 CUDA 可用性，不会自动切换设备。Linux CPU 模式仍安装锁定的 CUDA-enabled Torch（不用 GPU 也能计算），下载占用较大；本版没有宣称是精简 CPU 安装包。

Windows 若出现 Application Control / WDAC / AppLocker 阻止 Python 的提示，请使用组织批准的 Python/运行环境，或请管理员批准相关可执行文件。不要重命名、复制解释器或关闭策略来绕过限制。Notebook 启动器不能解除系统限制。

## 私有资产登录

在当前环境的终端使用交互式 `hf auth login`，按提示提供具有读取权限的 HF token。若 `hf` 未加入 PATH，setup 后可直接使用应用环境的 CLI：

```bash
# Linux / Colab runtime terminal
.venv/bin/hf auth login
```

```powershell
# Windows PowerShell
.\.venv\Scripts\hf.exe auth login
```

这是 Hugging Face 的正常凭据存储，与 OpenAI key 分开。Notebook 不接收 HF token 参数；不要把它写进单元格或 Git。现有完整资产包不需要联网登录。

## 查询与密钥

查询格默认 `RUN_QUERY=False`，运行全部单元格不会自动产生 OpenAI 费用。准备好后改为 `True`，输入 3–2000 字符的英文问题，并使用隐藏输入提供 API key。支持 `gpt-4.1`、`gpt-5`、`gpt-5.6-luna`，原始问题会原样提交。

客户端只向本机地址 POST 一次，随后 GET 轮询至 `completed`、`abstained`、`failed` 或 `cancelled`。key 不写进代码、环境变量、命令行或文件；错误响应不回显请求体，展示结果前会脱敏，最后释放 Notebook 的 key 变量。隐藏输入不可用时会停止，不退回明文输入。Notebook 内核与后台在处理请求时仍需要临时持有 key；释放变量并不等于安全擦除内存。

等待超时或中断客户端不会取消已经接受的后台请求；先检查状态，避免立即重试造成重复收费。结果与输出可能包含查询和建筑数据，分享或提交 Notebook 前请执行 **Clear All Outputs** 并保存。

## 不使用 Notebook 的等价命令

以下 `python` 指能被系统批准执行的 Python 3.10+；setup 会单独准备应用 Python 3.12。

```bash
python scripts/demo_launcher.py setup --device cpu
python scripts/demo_launcher.py assets --assets-dir assets
# 明确需要下载时才追加 --download
python scripts/demo_launcher.py start --assets-dir assets --device cpu --port 7860 --state-dir .demo-state
python scripts/demo_launcher.py status --state-dir .demo-state
python scripts/demo_launcher.py stop --state-dir .demo-state
```

脚本验证（只使用标准库与本地模拟 HTTP，不调用付费模型）：

```bash
python -m unittest discover -s scripts/tests -p test_notebook_client.py -v
```
