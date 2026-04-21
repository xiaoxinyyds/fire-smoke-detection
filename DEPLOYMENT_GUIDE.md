# 火灾烟雾检测系统 Streamlit 部署指南

本指南将帮助您将火灾烟雾检测系统部署到 Streamlit Cloud，实现互联网访问。

## 目录
1. [部署方案选择](#部署方案选择)
2. [部署准备](#部署准备)
3. [部署到 Streamlit Community Cloud](#部署到-streamlit-community-cloud)
4. [本地网络分享（备用方案）](#本地网络分享备用方案)
5. [注意事项与故障排除](#注意事项与故障排除)

## 部署方案选择

### 方案一：Streamlit Community Cloud（推荐）
- **优点**：免费、简单、无需服务器维护
- **限制**：CPU环境、有限的计算资源、应用休眠机制
- **适用场景**：演示、小型项目、测试部署

### 方案二：本地网络分享
- **优点**：完全控制、可使用GPU、适合开发调试
- **限制**：需要稳定网络、本地电脑需保持开机
- **工具**：ngrok、localhost.run、Cloudflare Tunnel

### 方案三：自托管服务器
- **优点**：完全控制、性能可扩展
- **限制**：需要服务器、维护成本高
- **平台**：AWS、Google Cloud、Azure、VPS

## 部署准备

### 1. 代码优化
确保您的 `app.py` 已针对部署环境优化：
- 所有路径使用相对路径 ✅ 已完成
- 有文件存在检查 ✅ 已完成
- 支持纯CPU环境 ✅ 已完成
- 使用ultralytics-opencv-headless避免GUI依赖 ✅ 已完成

### 2. 依赖管理
已创建两个依赖文件：
- `requirements.txt`：基础依赖
- `requirements_deploy.txt`：部署推荐依赖（更完整，使用ultralytics-opencv-headless避免GUI依赖）
- `packages.txt`：系统依赖（备选方案，用于安装libGL等系统库，如果`ultralytics-opencv-headless`仍需要）

### 3. 配置文件
已创建 `.streamlit/config.toml`，包含：
- 主题配置
- 服务器设置（最大上传50MB）
- 安全配置

### 4. 必要文件
确保以下文件存在于仓库中：
```
├── app.py                          # 主应用文件
├── requirements.txt               # 基础依赖文件
├── requirements_deploy.txt        # 部署专用依赖（推荐使用）
├── packages.txt                   # 系统依赖（备选，用于解决libGL错误，如不需要可删除）
├── .streamlit/config.toml         # Streamlit配置
├── runtime.txt                    # Python版本指定
├── runs/detect/fire_smoke_optimized1/
│   ├── weights/best.pt           # 模型文件（必须）
│   ├── weights/last.pt           # 备用模型
│   ├── results.csv               # 训练指标（可选）
│   └── *.png                     # 训练图表（可选）
└── test1.png                     # 测试图片（可选）
```

**注意**：模型文件约22MB，GitHub允许单文件≤100MB，可以正常提交。

## 部署到 Streamlit Community Cloud

### 步骤1：创建 GitHub 仓库
1. 登录 [GitHub](https://github.com)
2. 点击右上角 "+" → "New repository"
3. 填写仓库信息：
   - Repository name: `fire-smoke-detection`（或其他名称）
   - Description: "基于YOLOv8的火灾烟雾检测系统"
   - 选择 Public（公开仓库）
   - 初始化 README（可选）
4. 点击 "Create repository"

### 步骤2：上传代码到 GitHub
#### 方法A：使用Git命令行
```bash
# 初始化Git仓库
git init
git add .
git commit -m "Initial commit: Fire smoke detection system"

# 添加远程仓库
git remote add origin https://github.com/你的用户名/仓库名.git
git branch -M main
git push -u origin main
```

#### 方法B：使用GitHub Desktop 或网页上传
将以下文件上传到仓库：
- `app.py`
- `requirements.txt`（或`requirements_deploy.txt`，**推荐使用`requirements_deploy.txt`并重命名为`requirements.txt`**）
- `packages.txt`（系统依赖，解决libGL错误，如不需要可删除）
- `.streamlit/` 文件夹
- `runtime.txt`
- `runs/` 文件夹（包含模型文件）
- 其他必要文件

### 步骤3：部署到 Streamlit Cloud
1. 访问 [Streamlit Community Cloud](https://share.streamlit.io/)
2. 使用GitHub账号登录
3. 点击 "New app"
4. 配置应用：
   - **Repository**: 选择您的仓库
   - **Branch**: `main`
   - **Main file path**: `app.py`
   - **Python version**: 3.9（自动从runtime.txt读取）
   - **依赖文件**: Streamlit Cloud会自动使用仓库根目录的 `requirements.txt` 文件。确保该文件使用 `ultralytics-opencv-headless`（或确保 `requirements_deploy.txt` 已重命名为 `requirements.txt`）
5. 点击 "Deploy!"

### 步骤4：等待部署完成
- Streamlit会自动安装依赖并启动应用
- **注意**：如果遇到系统依赖安装错误（如`libgl1-mesa-glx`不可用）：
  1. 检查`packages.txt`文件中的包名是否适用于当前系统（Debian Trixie）
  2. 如果错误持续，可以删除`packages.txt`文件，仅依赖`ultralytics-opencv-headless`包
  3. 重新推送代码并重新部署
- 部署时间：首次部署约3-5分钟
- 部署成功后，会获得一个URL：`https://你的应用名.streamlit.app`

### 步骤5：访问与测试
1. 打开部署URL
2. 测试图片上传功能
3. 测试摄像头检测（需要浏览器摄像头权限）
4. 验证所有功能正常工作

## 本地网络分享（备用方案）

### 使用 ngrok（推荐）
1. **安装 ngrok**
   ```bash
   # 访问 https://ngrok.com/ 注册并下载
   # 或使用包管理器安装
   ```

2. **启动本地Streamlit应用**
   ```bash
   streamlit run app.py --server.port 8501
   ```

3. **创建隧道**
   ```bash
   ngrok http 8501
   ```

4. **访问公共URL**
   - ngrok会提供一个类似 `https://xxxx.ngrok.io` 的URL
   - 该URL可在任何设备上访问

### 使用 localhost.run
```bash
ssh -R 80:localhost:8501 ssh.localhost.run
```

### 使用 Cloudflare Tunnel
1. 安装 Cloudflare Tunnel
2. 配置隧道指向本地8501端口
3. 获得Cloudflare提供的域名

## 注意事项与故障排除

### 常见问题
1. **模型加载失败**
   ```
   错误：FileNotFoundError: [Errno 2] No such file or directory: 'runs/detect/...'
   ```
   **解决方案**：确保模型文件路径正确，且已提交到Git仓库。

2. **依赖安装失败**
   ```
   错误：Could not find a version that satisfies the requirement...
   ```
   **解决方案**：
   - 使用 `requirements_deploy.txt` 替代 `requirements.txt`
   - 移除版本号限制：`torch>=2.0.0` → `torch`

3. **libGL.so.1 缺失错误**
   ```
   错误：ImportError: libGL.so.1: cannot open shared object file: No such file or directory
   ```
   **原因**：OpenCV需要系统图形库，但Streamlit Cloud服务器缺少相关依赖。
   **解决方案**：
   - **方法一（推荐）**：使用 `ultralytics-opencv-headless` 包替代 `ultralytics`
     - 确保使用 `requirements_deploy.txt`（已配置为使用`ultralytics-opencv-headless`）
     - 或者修改 `requirements.txt` 将 `ultralytics>=8.2.0` 替换为 `ultralytics-opencv-headless>=8.2.0`
   
   - **方法二**：使用系统依赖补丁（备选方案）
     - 在仓库根目录创建 `packages.txt` 文件，添加以下内容（适用于Debian Trixie）：
       ```
       libgl1
       libglib2.0-0
       libsm6
       libxext6
       libxrender1
       libxfixes3
       libxi6
       libfontconfig1
       libfreetype6
       ```
     - **注意**：如果`libgl1`包不可用，可以尝试删除`packages.txt`文件，仅使用方法一
     - 确保使用 `requirements_deploy.txt`（包含`ultralytics-opencv-headless`无GUI版本）
   
   - 重新部署应用，Streamlit Cloud会自动安装系统依赖

4. **内存不足**
   ```
   错误：Killed - 应用崩溃
   ```
   **解决方案**：
   - Streamlit Cloud免费版内存有限（约1GB）
   - 优化模型：使用更小的YOLOv8n模型
   - 减少并发用户数

5. **摄像头不可用**
   - Streamlit Cloud无法访问本地摄像头
   - 解决方案：仅使用图片上传功能，或提示用户使用本地部署

### 性能优化建议
1. **模型优化**
   ```python
   # 在app.py中强制使用CPU
   device = "cpu"
   ```

2. **缓存优化**
   ```python
   @st.cache_resource
   def load_model():
       # 模型加载代码
       pass
   ```

3. **图片处理优化**
   - 限制上传图片大小
   - 使用缩略图预览

### 安全注意事项
1. **公开访问**
   - Streamlit Cloud应用默认公开
   - 如需隐私，可设置密码保护（付费功能）

2. **文件上传**
   - 已配置最大50MB限制
   - 建议添加文件类型验证

3. **API密钥**
   - 不要在代码中硬编码敏感信息
   - 使用Streamlit Secrets管理环境变量

### 维护与更新
1. **更新代码**
   - 提交更改到GitHub
   - Streamlit Cloud会自动重新部署

2. **查看日志**
   - 在Streamlit Cloud仪表板查看应用日志
   - 监控错误和性能问题

3. **版本管理**
   - 使用Git标签管理版本
   - 保持依赖版本更新

## 总结
通过Streamlit Cloud，您可以在几分钟内将火灾烟雾检测系统部署到互联网。虽然免费版本有资源限制，但对于演示和小型项目完全足够。如需更强大功能，可考虑升级到付费计划或使用其他部署方案。

**部署成功的关键**：
1. ✅ 正确的文件结构
2. ✅ 完整的依赖列表（包括Python依赖和可选的系统依赖）
3. ✅ 模型文件包含在仓库中
4. ✅ 使用相对路径
5. ✅ 考虑CPU环境优化
6. ✅ 使用ultralytics-opencv-headless避免GUI依赖

如有问题，请参考：
- [Streamlit文档](https://docs.streamlit.io/)
- [Streamlit Cloud文档](https://docs.streamlit.io/streamlit-community-cloud)
- [GitHub Issues](https://github.com/your-repo/issues)