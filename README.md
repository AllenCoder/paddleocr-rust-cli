# ocr-rust (PaddleOCR Rust 纯本地离线单文件化 OCR 推理命令行工具)

这是一个基于 **Rust** 与 **ONNX Runtime (ort)** 编写的纯本地离线 OCR 推理 CLI 工具。

本项目将 PaddleOCR 的**文本检测 (DBNet)** 与**文本识别 (CRNN)** 全链路用 Rust 进行了重写，不依赖任何臃肿且难以跨平台配置的 C++ OpenCV 动态库，图片处理与轮廓寻找均采用 100% 纯 Rust 库实现。

为了实现 100% 独立闭环、无需依赖任何外部模型及字典文件的单文件运行体验，**项目将默认的模型二进制和 6904 字离线字典直接静态嵌入 (Embed) 到了可执行文件中**。

---

## 📦 项目特色

1. **🚀 极速本地推理**：直接调用微软官方 `onnxruntime` 底层 C API 编译优化，比浏览器 WASM 推理速度提升约 1.5x - 3x。
2. **📦 纯 Rust 图片处理**：使用 `image` 和 `imageproc` 进行图像预处理、归一化与二值化轮廓提取，免除 CMake 依赖，保证 100% 绿色编译。
3. **🔋 单二进制纯绿色运行**：
   - 模型二进制和中文字典**完全内联编译到 `ocr-rust.exe` 中**。
   - 您可以将 `ocr-rust.exe` 和 `onnxruntime.dll` 拷贝到**任意磁盘路径或任意无代码的干净机器上直接执行**，无需附带任何同级或父级的 `models/` 文件夹！
   - 仍然支持通过命令行参数自适应指定并加载其他自定义的外部模型与字典。

---

## 🏃 快速开始 (即开即用)

您不需要安装任何 Rust 或 C++ 环境，只需直接运行。**默认情况下，程序会保持安静，并直接以 JSON 数据结构输出识别的框坐标和文本结果，极其便于第三方程序管道调用**：

```bash
# 1. 拷贝 ocr-rust.exe 和 onnxruntime.dll 到任意同一目录下
# 2. 传入要识别的图片路径 (位置参数，直接在结尾写路径即可，不需要带任何参数前缀)
.\ocr-rust.exe C:\path\to\your\image.jpg
```

输出示例 (JSON)：
```json
[
  {
    "box": [93, 128, 122, 11],
    "text": "文档图片可进化"
  },
  {
    "box": [781, 521, 159, 12],
    "text": "支持多种语言和场景"
  }
]
```

### ⚙️ 性能与辅助调试信息开关 (`-i, --info`)

如果您想输出详细的引擎载入、预处理、DBNet 检测、CRNN 识别的微秒级耗时及每一步骤的运行状态，请加上 **`-i`** 或 **`--info`** 标志开关：

```bash
.\ocr-rust.exe C:\path\to\your\image.jpg -i
```

### ⚙️ 进阶自定义资源参数

程序仍然支持加载您自己解包或指定的外部 ONNX 模型及字典文件：

```bash
.\ocr-rust.exe <您的图片路径> [-d <自定义检测模型路径>] [-r <自定义识别模型路径>] [--dict <自定义字典路径>]
```

参数使用规则：
- `-d, --det-model`：若未指定该参数，默认自动从二进制内存中载入内嵌的检测模型。
- `-r, --rec-model`：若未指定该参数，默认自动从二进制内存中载入内嵌的识别模型。
- `--dict`：若未指定该参数，默认自动载入内嵌的 6904 字全中文字典。
- `-h, --help`：打印命令行的辅助参数提示和详细帮助手册。

---

## 🔨 二次开发与编译指南

如果您对核心的推理与 CTC 解码算法进行了修改，或者想替换默认嵌入的模型，您可以重新编译：

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

编译完成后，最新的静态内联版 `ocr-rust.exe` 将生成在 `target/release/` 目录中。
