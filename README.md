# ocr-rust (PaddleOCR Rust 纯本地离线 OCR 推理命令行工具)

这是一个基于 **Rust** 与 **ONNX Runtime (ort)** 编写的纯本地离线 OCR 推理 CLI 工具。

本项目将 PaddleOCR 的**文本检测 (DBNet)** 与**文本识别 (CRNN)** 全链路用 Rust 进行了重写，不依赖任何臃肿且难以跨平台配置的 C++ OpenCV 动态库，图片处理与轮廓寻找均采用 100% 纯 Rust 库实现。

为了实现 100% 独立闭环、免配置即开即用，**项目仓库中已经内置了 Windows x64 下的所有运行依赖文件和轻量化 ONNX 模型**。

---

## 📦 项目特色

1. **🚀 极速本地推理**：直接调用微软官方 `onnxruntime` 底层 C API 编译优化，比浏览器 WASM 推理速度提升约 1.5x - 3x。
2. **📦 纯 Rust 图片处理**：使用 `image` 和 `imageproc` 进行图像预处理、归一化与二值化轮廓提取，免除 CMake 依赖，保证 100% 绿色编译。
3. **🔧 零配置开箱即用**：
   - 仓库同级目录内置了编译好的可执行程序 `ocr-rust.exe` 以及运行所需的 `onnxruntime.dll`。
   - 内置了极速版中英文检测/识别 ONNX 模型。
   - 内置了从模型配置中完美解析提炼的 6904 字离线中文字典。

---

## 🏃 快速开始 (即开即用)

您不需要安装任何 Rust 或 C++ 环境，只需在 Windows 终端中直接运行：

```bash
# 识别同级目录下的默认测试图片 test.jpg
.\ocr-rust.exe -i test.jpg
```

### ⚙️ 进阶参数

程序提供了自适应参数支持：

```bash
.\ocr-rust.exe -i <您的图片路径> -d <检测模型路径> -r <识别模型路径> --dict <中文字典路径>
```

默认参数值（缺省时自动使用）：
- `-d, --det-model`: `models/PP-OCRv6_tiny_det_onnx_infer/inference.onnx`
- `-r, --rec-model`: `models/PP-OCRv6_tiny_rec_onnx_infer/inference.onnx`
- `--dict`: `models/dict.txt`

---

## 📋 运行输出预期

```text
🔔 正在载入文本检测模型: "models/PP-OCRv6_tiny_det_onnx_infer/inference.onnx"
🔔 正在载入文本识别模型: "models/PP-OCRv6_tiny_rec_onnx_infer/inference.onnx"
📸 正在读取图片: "test.jpg"
🔍 正在提取文本区域...
🎯 检测到 26 个文本区域，开始执行识别...
  👉 [框 4] 坐标:(93,128,122,11) -> 识别结果: "文档图片可进化"
  👉 [框 5] 坐标:(770,183,205,11) -> 识别结果: "本地识别总耗时：120ms!"
  👉 [框 8] 坐标:(1061,279,141,12) -> 识别结果: "当前状态：运行中"
  👉 [框 23] 坐标:(781,521,159,12) -> 识别结果: "支持多种语言和场景"
✨ OCR 任务处理完成！
```

---

## 🔨 二次开发与编译指南

如果您对核心的推理与 CTC 解码算法进行了修改，想要重新编译它：

### 1. 编译依赖准备
在 Windows 11 环境下，编译需要链接本地 ONNX Runtime 导出库。本仓库的 `libs/` 目录下已放置了所需的 `onnxruntime.lib` 与 `onnxruntime.dll`。

### 2. 执行编译
在 Powershell 或 CMD 中，强行指定 `ORT_DYLIB_PATH` 环境变量并进行 release 编译：

```powershell
# 1. 载入 MSVC 命令行工具环境
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64

# 2. 设置 DLL 路径并执行 release 编译
set ORT_DYLIB_PATH=C:\Users\shuangshuang.wangs_1.WUYING\ocr-rust\libs\onnxruntime.dll
cargo build --release
```

编译完成后，最新的 `ocr-rust.exe` 将生成在 `target/release/` 目录中。

