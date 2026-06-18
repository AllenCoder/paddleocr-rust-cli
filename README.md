# ocr-rust (PaddleOCR Rust 离线命令行工具)

这是一个基于 **Rust** 与 **ONNX Runtime (ort)** 编写的纯本地离线 OCR 推理 CLI 工具。

本项目将 PaddleOCR 的**文本检测 (DBNet)** 与**文本识别 (CRNN)** 全链路用 Rust 进行了重写，不依赖任何臃肿且难以跨平台配置的 C++ OpenCV 动态库，图片处理与轮廓寻找均采用 100% 纯 Rust 库实现。

---

## 🛠️ 项目特色

1. **🚀 极速本地推理**：直接调用微软官方 `onnxruntime` 底层 C API 编译优化，比浏览器 WASM 推理速度提升约 1.5x - 3x。
2. **📦 纯 Rust 图片处理**：使用 `image` 和 `imageproc` 进行图像归一化预处理与二值化轮廓提取，避免了 C++ 动态链接编译报错的硬伤，保证 100% 可编译。
3. **⚙️ 资源轻量**：编译打包出的二进制文件仅约 **10MB - 20MB** 左右（不含模型），适合作为 CLI 系统工具集成。

---

## 📂 编译与运行准备

### 1. 提取 ONNX 静态模型文件
PaddleOCR-js 的离线模型采用 `.tar` 打包。您需要提前将其中的 `.onnx` 格式推理模型提取出来供 Rust 项目加载：

```bash
# 1. 提取检测模型 (inference.onnx)
tar -xf ../ocr-demo/public/models/PP-OCRv6_tiny_det_onnx_infer.tar
# 会解压出: PP-OCRv6_tiny_det_onnx_infer/inference.onnx

# 2. 提取识别模型 (inference.onnx)
tar -xf ../ocr-demo/public/models/PP-OCRv6_tiny_rec_onnx_infer.tar
# 会解压出: PP-OCRv6_tiny_rec_onnx_infer/inference.onnx
```

### 2. 准备中文字典文件
您可以直接将 PaddleOCR 官方字典（例如 `ppocr_keys_v1.txt`）复制到该目录下。如果未提供字典，程序会自动回退到数字、英文字符集的小字库进行最小化演示。

---

## 🔨 编译指南

在配置好 Rust 工具链 (`rustup` / `cargo`) 的开发机上，执行以下命令：

```bash
# 编译并以 Release 模式打包
cargo build --release
```

> 💡 **关于 ONNX Runtime 的动态库 (`onnxruntime.dll` / `.so`)**：
> 本项目的 `Cargo.toml` 中配置了 `ort` 库的 `copy-dylibs` 特性。当您首次运行 `cargo build` 或 `cargo run` 时，库会自动从微软官方分发节点下载对应平台的 `onnxruntime` 并自动复制到您的 `target/debug` 或 `target/release` 输出目录中，您无需手动下载和配置 DLL！

---

## 🏃 运行使用示例

编译成功后，可以直接通过 Cargo 或者生成的二进制程序进行 OCR 识别测试：

```bash
cargo run -- \
  --image test.jpg \
  --det_model ./PP-OCRv6_tiny_det_onnx_infer/inference.onnx \
  --rec_model ./PP-OCRv6_tiny_rec_onnx_infer/inference.onnx \
  --dict ./ppocr_keys_v1.txt
```

### 📋 运行输出预期

运行后，您的命令行控制台将输出识别结果明细：
```text
🔔 正在载入文本检测模型: "PP-OCRv6_tiny_det_onnx_infer/inference.onnx"
🔔 正在载入文本识别模型: "PP-OCRv6_tiny_rec_onnx_infer/inference.onnx"
📸 正在读取图片: "test.jpg"
🔍 正在提取文本区域...
🎯 检测到 3 个文本区域，开始执行识别...
  👉 [框 1] 坐标:(45,12,120,32) -> 识别结果: "迈迎运近"
  👉 [框 2] 坐标:(200,10,80,30) -> 识别结果: "离线测试"
  👉 [框 3] 坐标:(12,80,150,28) -> 识别结果: "PaddleOCR"
✨ OCR 任务处理完成！
```
