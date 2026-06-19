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
    #[arg(long)]
    dict: Option<PathBuf>,

    /// 将 OCR 完整结果 (JSON) 保存到指定的本地文件中
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// 追加记录耗时与识别概况的日志文件路径 [可选，默认在当前目录下追加记录到 ocr.log]
    #[arg(short, long)]
    log: Option<PathBuf>,

    /// 控制台输出格式：text (人类友好简明列表，超长折叠) 或 json (完整 JSON 数据)
    #[arg(short, long, default_value = "text")]
    format: String,

    /// 是否输出详细性能耗时分析与步骤辅助信息到 stderr
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

fn get_current_time_string() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    if let Ok(duration) = SystemTime::now().duration_since(UNIX_EPOCH) {
        let secs = duration.as_secs();
        // 简单的 Unix 时间戳转东八区北京时间 (YYYY-MM-DD HH:MM:SS)
        let seconds_in_day = 86400;
        let days = secs / seconds_in_day;
        let seconds_of_day = secs % seconds_in_day;

        let hour = (seconds_of_day / 3600 + 8) % 24; // 转换到 UTC+8
        let minute = (seconds_of_day % 3600) / 60;
        let second = seconds_of_day % 60;

        let mut year = 1970;
        let mut day_of_year = days;

        loop {
            let is_leap = (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
            let days_in_year = if is_leap { 366 } else { 365 };
            if day_of_year >= days_in_year {
                day_of_year -= days_in_year;
                year += 1;
            } else {
                break;
            }
        }

        let is_leap = (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
        let month_days = [31, if is_leap { 29 } else { 28 }, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
        let mut month = 1;
        for &d in &month_days {
            if day_of_year >= d as u64 {
                day_of_year -= d as u64;
                month += 1;
            } else {
                break;
            }
        }
        let day = day_of_year + 1;

        format!("{:04}-{:02}-{:02} {:02}:{:02}:{:02}", year, month, day, hour, minute, second)
    } else {
        "Unknown Time".to_string()
    }
}

fn configure_linux_runtime() {
    #[cfg(target_os = "linux")]
    {
        for (key, value) in [
            ("OMP_NUM_THREADS", "1"),
            ("KMP_NUM_THREADS", "1"),
            ("GOMP_NUM_THREADS", "1"),
            ("OMP_WAIT_POLICY", "PASSIVE"),
        ] {
            if std::env::var_os(key).is_none() {
                std::env::set_var(key, value);
            }
        }
    }
}

fn build_session_builder() -> ort::Result<ort::session::builder::SessionBuilder> {
    let intra_threads = if cfg!(target_os = "linux") { 1 } else { 4 };

    let builder = Session::builder()?.with_intra_threads(intra_threads)?;

    #[cfg(target_os = "linux")]
    let builder = builder
        .with_parallel_execution(false)?
        .with_inter_threads(1)?
        .with_intra_op_spinning(false)?
        .with_inter_op_spinning(false)?;

    Ok(builder)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let show_info = args.info;
    configure_linux_runtime();

    // 自动推断并配置 ORT_DYLIB_PATH 环境变量以支持动态加载
    if std::env::var("ORT_DYLIB_PATH").is_err() {
        if let Ok(exe_path) = std::env::current_exe() {
            if let Some(exe_dir) = exe_path.parent() {
                #[cfg(target_os = "windows")]
                let lib_name = "onnxruntime.dll";
                #[cfg(target_os = "macos")]
                let lib_name = "libonnxruntime.dylib";
                #[cfg(target_os = "linux")]
                let lib_name = "libonnxruntime.so";

                #[cfg(any(target_os = "windows", target_os = "macos", target_os = "linux"))]
                {
                    // 1. 尝试当前可执行文件同级目录
                    let local_lib = exe_dir.join(lib_name);
                    if local_lib.exists() {
                        std::env::set_var("ORT_DYLIB_PATH", local_lib.to_string_lossy().to_string());
                    } else {
                        // 2. 尝试当前可执行文件目录下的 libs 子目录
                        let libs_dir = exe_dir.join("libs").join(lib_name);
                        if libs_dir.exists() {
                            std::env::set_var("ORT_DYLIB_PATH", libs_dir.to_string_lossy().to_string());
                        }
                    }
                }
            }
        }
    }

    use std::fs::OpenOptions;
    use std::io::Write;

    let format_arg = args.format.to_lowercase();
    if format_arg != "text" && format_arg != "json" {
        eprintln!("❌ 错误: --format 参数仅支持 \"text\" 或 \"json\"");
        std::process::exit(1);
    }

    let start_total = std::time::Instant::now();

    // 1. 初始化 ONNX Runtime 环境并载入模型
    let start_load = std::time::Instant::now();
    let mut det_session = if let Some(ref path) = args.det_model {
        let resolved = resolve_path(path.clone());
        if show_info {
            eprintln!("🔔 正在载入外部文本检测模型: {:?}", resolved);
        }
        build_session_builder()?.commit_from_file(&resolved)?
    } else {
        if show_info {
            eprintln!("🔔 正在载入内嵌默认文本检测模型 (PP-OCRv6 tiny)");
        }
        build_session_builder()?.commit_from_memory(DEFAULT_DET_MODEL)?
    };

    let mut rec_session = if let Some(ref path) = args.rec_model {
        let resolved = resolve_path(path.clone());
        if show_info {
            eprintln!("🔔 正在载入外部文本识别模型: {:?}", resolved);
        }
        build_session_builder()?.commit_from_file(&resolved)?
    } else {
        if show_info {
            eprintln!("🔔 正在载入内嵌默认文本识别模型 (PP-OCRv6 tiny)");
        }
        build_session_builder()?.commit_from_memory(DEFAULT_REC_MODEL)?
    };
    let load_duration = start_load.elapsed();

    // 2. 加载图片与预处理
    let start_preprocess = std::time::Instant::now();
    if show_info {
        eprintln!("📸 正在读取图片: {:?}", args.image);
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
        eprintln!("🔍 正在提取文本区域...");
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
        eprintln!("🎯 检测到 {} 个文本区域，开始执行识别...", boxes.len());
    }

    // 6. 加载字典
    let dict = if let Some(ref path) = args.dict {
        let resolved = resolve_path(path.clone());
        if show_info {
            eprintln!("🔔 正在载入外部字典: {:?}", resolved);
        }
        load_dict(&resolved)?
    } else {
        if show_info {
            eprintln!("🔔 正在载入内嵌默认中文字典");
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
            eprintln!("  👉 [框 {}] 坐标:({},{},{},{}) -> 识别结果: \"{}\"", i + 1, bx, by, bw, bh, text);
        }
        results.push((bx, by, bw, bh, text));
    }
    let rec_duration = start_rec.elapsed();
    let total_duration = start_total.elapsed();

    // 8. 构造完整 JSON 结果 (包含性能指标和识别内容)
    let mut json_items = Vec::new();
    for &(bx, by, bw, bh, ref text) in &results {
        let escaped_text = text.replace('\\', "\\\\").replace('\"', "\\\"");
        json_items.push(format!(
            "    {{\n      \"box\": [{}, {}, {}, {}],\n      \"text\": \"{}\"\n    }}",
            bx, by, bw, bh, escaped_text
        ));
    }
    let results_json = json_items.join(",\n");
    let json_str = format!(
        "{{\n  \"metrics\": {{\n    \"load_model_ms\": {:.2},\n    \"preprocess_ms\": {:.2},\n    \"det_inference_ms\": {:.2},\n    \"rec_inference_ms\": {:.2},\n    \"total_ms\": {:.2}\n  }},\n  \"results\": [\n{}\n  ]\n}}",
        load_duration.as_secs_f64() * 1000.0,
        preprocess_duration.as_secs_f64() * 1000.0,
        det_duration.as_secs_f64() * 1000.0,
        rec_duration.as_secs_f64() * 1000.0,
        total_duration.as_secs_f64() * 1000.0,
        results_json
    );

    // 9. 处理输出逻辑
    if let Some(ref out_path) = args.output {
        // 保存 JSON 到本地文件
        let mut file = File::create(out_path)?;
        file.write_all(json_str.as_bytes())?;
        
        // 在终端打印清爽的成功摘要，不刷屏
        println!("✨ OCR 任务处理完成！已成功识别 {} 个文本区域，结果已保存至 \"{}\" (总耗时: {:.2}ms)", 
            results.len(), 
            out_path.display(),
            total_duration.as_secs_f64() * 1000.0
        );
    } else {
        // 如果没有指定输出文件，根据 format 输出
        if format_arg == "json" {
            println!("{}", json_str);
        } else {
            // text 格式：如果结果很多则折叠防刷屏
            let max_display = 20;
            if results.len() <= max_display {
                for (i, &(bx, by, bw, bh, ref text)) in results.iter().enumerate() {
                    println!("[{:02}] (x:{}, y:{}, w:{}, h:{}) -> \"{}\"", i + 1, bx, by, bw, bh, text);
                }
            } else {
                for (i, &(bx, by, bw, bh, ref text)) in results.iter().take(10).enumerate() {
                    println!("[{:02}] (x:{}, y:{}, w:{}, h:{}) -> \"{}\"", i + 1, bx, by, bw, bh, text);
                }
                println!("... (已省略中间 {} 个文本区域，您可以使用 -o/--output <FILE> 保存完整 JSON 结果) ...", results.len() - 20);
                for (i, &(bx, by, bw, bh, ref text)) in results.iter().skip(results.len() - 10).enumerate() {
                    println!("[{:02}] (x:{}, y:{}, w:{}, h:{}) -> \"{}\"", results.len() - 10 + i + 1, bx, by, bw, bh, text);
                }
            }
        }
    }

    // 10. 追加记录性能指标到日志文件
    let log_path = args.log.clone().unwrap_or_else(|| PathBuf::from("ocr.log"));
    if let Ok(mut log_file) = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
    {
        let time_str = get_current_time_string();
        let log_line = format!(
            "[{}] Image: {:?} | Regions: {} | Load: {:.2}ms | Preprocess: {:.2}ms | Det: {:.2}ms | Rec: {:.2}ms | Total: {:.2}ms\n",
            time_str,
            args.image,
            results.len(),
            load_duration.as_secs_f64() * 1000.0,
            preprocess_duration.as_secs_f64() * 1000.0,
            det_duration.as_secs_f64() * 1000.0,
            rec_duration.as_secs_f64() * 1000.0,
            total_duration.as_secs_f64() * 1000.0,
        );
        let _ = log_file.write_all(log_line.as_bytes());
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
