#!/bin/bash
set -e

echo "📦 Installing system build dependencies..."
sudo apt-get update
sudo apt-get install -y build-essential pkg-config libssl-dev curl tar

echo "📥 Downloading official ONNX Runtime v1.18.1 for Linux x64..."
curl -L -o /tmp/onnxruntime.tgz https://github.com/microsoft/onnxruntime/releases/download/v1.18.1/onnxruntime-linux-x64-1.18.1.tgz
tar -xzf /tmp/onnxruntime.tgz -C /tmp

echo "⚙️  Installing ONNX Runtime shared library and headers to /usr/lib and /usr/include..."
sudo cp -P /tmp/onnxruntime-linux-x64-1.18.1/lib/libonnxruntime.so* /usr/lib/
sudo cp -r /tmp/onnxruntime-linux-x64-1.18.1/include/* /usr/include/

echo "🧹 Cleaning up temporary files..."
rm -rf /tmp/onnxruntime*

echo "✅ Environment initialization completed successfully!"
rustc --version
