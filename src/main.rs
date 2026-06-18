use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use clap::Parser;
use image::{DynamicImage, GenericImageView, ImageBuffer, Luma};
use ndarray::{s, Array, Array4};
use ort::session::Session;

const DEFAULT_DET_MODEL: &[u8] = include_bytes!("../models/PP-OCRv6_tiny_det_onnx_infer/inference.onnx");
const DEFAULT_REC_MODEL: &[u8] = include_bytes!("../models/PP-OCRv6_tiny_rec_onnx_infer/inference.onnx");
const DEFAULT_DICT: &str = include_str!("../models/dict.txt");

#[derive(Parser, Debug)]
#[command(author, version, about = "PaddleOCR Standalone Rust Inference CLI Tool")]
struct Args {
    /// 输入要进行识别的测试图像路径 (位置参数)
    image: PathBuf,

    /// 文本检测模型 (inference.onnx) 路径 [可选，缺省则使用内嵌模型]
    #[arg(short, long)]
    det_model: Option<PathBuf>,

    /// 文本识别模型 (inference.onnx) 路径 [可选，缺省则使用内嵌模型]
    #[arg(short, long)]
    rec_model: Option<PathBuf>,

    /// 中文字典文本路径 [可选，缺省则使用内嵌中文字典]
    #[arg(short, long)]
    dict: Option<PathBuf>,

    /// 是否输出详细性能耗时分析与步骤辅助信息
    #[arg(short, long)]
    info: bool,
}

fn resolve_path(path: PathBuf) -> PathBuf {
    // 1. 尝试相对于当前工作目录解析并转为绝对路径
    if path.exists() {
        if let Ok(current_dir) = std::env::current_dir() {
            let abs_path = current_dir.join(&path);
            if abs_path.exists() {
                return abs_path;
            }
        }
        return path;
    }

    // 2. 尝试相对于可执行文件所在目录及其祖先目录级联解析
    if let Ok(exe_path) = std::env::current_exe() {
        let mut check_dir = exe_path.parent();
        for _ in 0..3 {
            if let Some(dir) = check_dir {
                let resolved = dir.join(&path);
                if resolved.exists() {
                    return resolved;
                }
                check_dir = dir.parent();
            } else {
                break;
            }
        }
    }

    path
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let show_info = args.info;

    let start_total = std::time::Instant::now();

    // 1. 初始化 ONNX Runtime 环境并载入模型
    let start_load = std::time::Instant::now();
    let mut det_session = if let Some(ref path) = args.det_model {
        let resolved = resolve_path(path.clone());
        if show_info {
            println!("🔔 正在载入外部文本检测模型: {:?}", resolved);
        }
        Session::builder()?
            .with_intra_threads(4)?
            .commit_from_file(&resolved)?
    } else {
        if show_info {
            println!("🔔 正在载入内嵌默认文本检测模型 (PP-OCRv6 tiny)");
        }
        Session::builder()?
            .with_intra_threads(4)?
            .commit_from_memory(DEFAULT_DET_MODEL)?
    };

    let mut rec_session = if let Some(ref path) = args.rec_model {
        let resolved = resolve_path(path.clone());
        if show_info {
            println!("🔔 正在载入外部文本识别模型: {:?}", resolved);
        }
        Session::builder()?
            .with_intra_threads(4)?
            .commit_from_file(&resolved)?
    } else {
        if show_info {
            println!("🔔 正在载入内嵌默认文本识别模型 (PP-OCRv6 tiny)");
        }
        Session::builder()?
            .with_intra_threads(4)?
            .commit_from_memory(DEFAULT_REC_MODEL)?
    };
    let load_duration = start_load.elapsed();

    // 2. 加载图片与预处理
    let start_preprocess = std::time::Instant::now();
    if show_info {
        println!("📸 正在读取图片: {:?}", args.image);
    }
    let img = image::open(&args.image)?;
    let (orig_w, orig_h) = img.dimensions();

    // 3. 检测模型图像预处理
    let det_size = 736;
    let (det_input, ratio_w, ratio_h) = preprocess_det(&img, det_size);
    let preprocess_duration = start_preprocess.elapsed();

    // 4. 执行文本检测推理 (DBNet)
    let start_det = std::time::Instant::now();
    let det_input_value = ort::value::Value::from_array(det_input.clone())?;
    let det_outputs = det_session.run(ort::inputs![det_input_value])?;
    let det_output_tensor = det_outputs[0].try_extract_array::<f32>()?;
    let prob_map = det_output_tensor.slice(s![0, 0, .., ..]);

    // 5. 检测后处理：二值化并提取文本区域轮廓
    if show_info {
        println!("🔍 正在提取文本区域...");
    }
    let mut binary_img: ImageBuffer<Luma<u8>, Vec<u8>> = ImageBuffer::new(det_size, det_size);
    for y in 0..det_size {
        for x in 0..det_size {
            let val = prob_map[[y as usize, x as usize]];
            let pixel_val = if val > 0.3 { 255 } else { 0 };
            binary_img.put_pixel(x, y, Luma([pixel_val]));
        }
    }

    let contours = imageproc::contours::find_contours(&binary_img);
    let mut boxes = Vec::new();

    for contour in contours {
        if contour.points.len() < 4 {
            continue;
        }
        let mut min_x = det_size;
        let mut max_x = 0;
        let mut min_y = det_size;
        let mut max_y = 0;
        for pt in &contour.points {
            if pt.x < min_x { min_x = pt.x; }
            if pt.x > max_x { max_x = pt.x; }
            if pt.y < min_y { min_y = pt.y; }
            if pt.y > max_y { max_y = pt.y; }
        }
        let w = max_x - min_x;
        let h = max_y - min_y;
        if w * h < 64 {
            continue;
        }
        let orig_min_x = (min_x as f32 / ratio_w) as u32;
        let orig_max_x = (max_x as f32 / ratio_w) as u32;
        let orig_min_y = (min_y as f32 / ratio_h) as u32;
        let orig_max_y = (max_y as f32 / ratio_h) as u32;
        boxes.push((orig_min_x, orig_min_y, orig_max_x - orig_min_x, orig_max_y - orig_min_y));
    }
    let det_duration = start_det.elapsed();

    if show_info {
        println!("🎯 检测到 {} 个文本区域，开始执行识别...", boxes.len());
    }

    // 6. 加载字典
    let dict = if let Some(ref path) = args.dict {
        let resolved = resolve_path(path.clone());
        if show_info {
            println!("🔔 正在载入外部字典: {:?}", resolved);
        }
        load_dict(&resolved)?
    } else {
        if show_info {
            println!("🔔 正在载入内嵌默认中文字典");
        }
        DEFAULT_DICT.lines().map(|line| line.to_string()).collect()
    };

    // 7. 遍历检测到的边框执行识别推理 (CRNN)
    let start_rec = std::time::Instant::now();
    let mut results = Vec::new();

    for (i, &(bx, by, bw, bh)) in boxes.iter().enumerate() {
        let crop_x = bx.min(orig_w - 1);
        let crop_y = by.min(orig_h - 1);
        let crop_w = bw.min(orig_w - crop_x);
        let crop_h = bh.min(orig_h - crop_y);
        if crop_w == 0 || crop_h == 0 {
            continue;
        }
        let cropped = img.crop_imm(crop_x, crop_y, crop_w, crop_h);
        let rec_h = 48;
        let rec_w = (crop_w as f32 * (rec_h as f32 / crop_h as f32)) as u32;
        let rec_input = preprocess_rec(&cropped, rec_w, rec_h);

        // 运行文本识别模型
        let rec_input_value = ort::value::Value::from_array(rec_input.clone())?;
        let rec_outputs = rec_session.run(ort::inputs![rec_input_value])?;
        let rec_tensor = rec_outputs[0].try_extract_array::<f32>()?;

        // 8. CTC 解码得到识别文本
        let text = decode_ctc(&rec_tensor, &dict);
        if show_info {
            println!("  👉 [框 {}] 坐标:({},{},{},{}) -> 识别结果: \"{}\"", i + 1, bx, by, bw, bh, text);
        }
        results.push((bx, by, bw, bh, text));
    }
    let rec_duration = start_rec.elapsed();
    let total_duration = start_total.elapsed();

    // 8. 输出结果
    if show_info {
        println!("\n⏱️ 详细性能分析与统计 (Verbose Metrics):");
        println!("  ⚡ 引擎载入与模型初始化: {:?}", load_duration);
        println!("  ⚡ 图像解码与预处理: {:?}", preprocess_duration);
        println!("  ⚡ 文本区域检测 (DBNet): {:?}", det_duration);
        println!("  ⚡ 文本片段识别 (CRNN): {:?}", rec_duration);
        println!("  ⏱️ 整体推理耗时: {:?}", total_duration);
        println!("  🎯 提取文本块总计: {} 个", results.len());
        println!("\n🔍 输出识别结果 (JSON):");
    }

    let mut json_items = Vec::new();
    for &(bx, by, bw, bh, ref text) in &results {
        let escaped_text = text.replace('\\', "\\\\").replace('\"', "\\\"");
        json_items.push(format!(
            "  {{\n    \"box\": [{}, {}, {}, {}],\n    \"text\": \"{}\"\n  }}",
            bx, by, bw, bh, escaped_text
        ));
    }
    let json_str = format!("[\n{}\n]", json_items.join(",\n"));
    println!("{}", json_str);

    if show_info {
        println!("✨ OCR 任务处理完成！");
    }
    Ok(())
}

/// 文本检测图像预处理 (归一化到 RGB 通道)
fn preprocess_det(img: &DynamicImage, target_size: u32) -> (Array4<f32>, f32, f32) {
    let (w, h) = img.dimensions();
    let ratio_w = target_size as f32 / w as f32;
    let ratio_h = target_size as f32 / h as f32;

    let resized = img.resize_exact(target_size, target_size, image::imageops::FilterType::Triangle);
    let mut array = Array::zeros((1, 3, target_size as usize, target_size as usize));

    for y in 0..target_size {
        for x in 0..target_size {
            let pixel = resized.get_pixel(x, y);
            // 按照 Det 的预处理均值与标准差进行 Normalize 运算
            let r = (pixel[0] as f32 / 255.0 - 0.485) / 0.229;
            let g = (pixel[1] as f32 / 255.0 - 0.456) / 0.224;
            let b = (pixel[2] as f32 / 255.0 - 0.406) / 0.225;

            array[[0, 0, y as usize, x as usize]] = r;
            array[[0, 1, y as usize, x as usize]] = g;
            array[[0, 2, y as usize, x as usize]] = b;
        }
    }

    (array, ratio_w, ratio_h)
}

/// 文本识别图像预处理 (高度固定为 48)
fn preprocess_rec(img: &DynamicImage, target_w: u32, target_h: u32) -> Array4<f32> {
    let resized = img.resize_exact(target_w, target_h, image::imageops::FilterType::Triangle);
    let mut array = Array::zeros((1, 3, target_h as usize, target_w as usize));

    for y in 0..target_h {
        for x in 0..target_w {
            let pixel = resized.get_pixel(x, y);
            // 按照 Rec 预处理的均值 (0.5) 和 标准差 (0.5) 进行归一化
            let r = (pixel[0] as f32 / 255.0 - 0.5) / 0.5;
            let g = (pixel[1] as f32 / 255.0 - 0.5) / 0.5;
            let b = (pixel[2] as f32 / 255.0 - 0.5) / 0.5;

            array[[0, 0, y as usize, x as usize]] = r;
            array[[0, 1, y as usize, x as usize]] = g;
            array[[0, 2, y as usize, x as usize]] = b;
        }
    }

    array
}

/// CTC 解码：从识别张量中映射字典，排除空白符并合并连续重复项
fn decode_ctc(tensor: &ndarray::ArrayViewD<f32>, dict: &[String]) -> String {
    let shape = tensor.shape();
    let steps = shape[1];
    let num_classes = shape[2];

    let mut result = String::new();
    let mut last_idx = -1;

    for t in 0..steps {
        let mut max_val = -f32::INFINITY;
        let mut max_idx = 0;

        for c in 0..num_classes {
            let val = tensor[[0, t, c]];
            if val > max_val {
                max_val = val;
                max_idx = c as i32;
            }
        }

        // 假设索引 0 为 CTC 的 Blank 占位空白符
        if max_idx > 0 && max_idx != last_idx {
            let char_idx = (max_idx - 1) as usize;
            if char_idx < dict.len() {
                result.push_str(&dict[char_idx]);
            }
        }
        last_idx = max_idx;
    }

    result
}

/// 加载中文字典
fn load_dict(path: &PathBuf) -> Result<Vec<String>, std::io::Error> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut dict = Vec::new();
    for line in reader.lines() {
        dict.push(line?);
    }
    Ok(dict)
}
