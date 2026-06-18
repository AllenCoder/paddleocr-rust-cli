use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use clap::Parser;
use image::{DynamicImage, GenericImageView, ImageBuffer, Luma};
use ndarray::{s, Array, Array4};
use ort::session::Session;

#[derive(Parser, Debug)]
#[command(author, version, about = "PaddleOCR Rust Inference Tool (Offline)")]
struct Args {
    /// 输入要识别的测试图像路径
    #[arg(short, long)]
    image: PathBuf,

    /// 文本检测检测模型 (inference.onnx) 路径
    #[arg(short, long, default_value = "models/PP-OCRv6_tiny_det_onnx_infer/inference.onnx")]
    det_model: PathBuf,

    /// 文本识别模型 (inference.onnx) 路径
    #[arg(short, long, default_value = "models/PP-OCRv6_tiny_rec_onnx_infer/inference.onnx")]
    rec_model: PathBuf,

    /// 中文字典密钥文本路径
    #[arg(short, long, default_value = "models/dict.txt")]
    dict: PathBuf,
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

    let det_model_path = resolve_path(args.det_model);
    let rec_model_path = resolve_path(args.rec_model);
    let dict_path = resolve_path(args.dict);

    // 1. 初始化 ONNX Runtime 环境并载入模型
    println!("🔔 正在载入文本检测模型: {:?}", det_model_path);
    let mut det_session = Session::builder()?
        .with_intra_threads(4)?
        .commit_from_file(&det_model_path)?;

    println!("🔔 正在载入文本识别模型: {:?}", rec_model_path);
    let mut rec_session = Session::builder()?
        .with_intra_threads(4)?
        .commit_from_file(&rec_model_path)?;

    // 2. 加载图片
    println!("📸 正在读取图片: {:?}", args.image);
    let img = image::open(&args.image)?;
    let (orig_w, orig_h) = img.dimensions();

    // 3. 检测模型图像预处理 (缩放到 32 的整数倍，这里假定 736x736 进行测试)
    let det_size = 736;
    let (det_input, ratio_w, ratio_h) = preprocess_det(&img, det_size);

    // 4. 执行文本检测推理 (DBNet)
    let det_input_value = ort::value::Value::from_array(det_input.clone())?;
    let det_outputs = det_session.run(ort::inputs![det_input_value])?;
    let det_output_tensor = det_outputs[0].try_extract_array::<f32>()?;
    
    // 获取概率图 shape: [1, 1, H, W]
    let prob_map = det_output_tensor.slice(s![0, 0, .., ..]);

    // 5. 检测后处理：二值化并提取文本区域轮廓
    println!("🔍 正在提取文本区域...");
    let mut binary_img: ImageBuffer<Luma<u8>, Vec<u8>> = ImageBuffer::new(det_size, det_size);
    for y in 0..det_size {
        for x in 0..det_size {
            let val = prob_map[[y as usize, x as usize]];
            let pixel_val = if val > 0.3 { 255 } else { 0 };
            binary_img.put_pixel(x, y, Luma([pixel_val]));
        }
    }

    // 利用 imageproc 寻找轮廓
    let contours = imageproc::contours::find_contours(&binary_img);
    let mut boxes = Vec::new();

    for contour in contours {
        // 过滤较小的噪点区域
        if contour.points.len() < 4 {
            continue;
        }

        // 计算包围盒
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

        // 过滤掉面积过小的框
        if w * h < 64 {
            continue;
        }

        // 还原回原图坐标尺寸
        let orig_min_x = (min_x as f32 / ratio_w) as u32;
        let orig_max_x = (max_x as f32 / ratio_w) as u32;
        let orig_min_y = (min_y as f32 / ratio_h) as u32;
        let orig_max_y = (max_y as f32 / ratio_h) as u32;

        boxes.push((orig_min_x, orig_min_y, orig_max_x - orig_min_x, orig_max_y - orig_min_y));
    }

    println!("🎯 检测到 {} 个文本区域，开始执行识别...", boxes.len());

    // 6. 加载字典
    let dict = load_dict(&dict_path)?;

    // 7. 遍历检测到的边框执行识别推理 (CRNN)
    for (i, &(bx, by, bw, bh)) in boxes.iter().enumerate() {
        // 确保裁剪边界安全
        let crop_x = bx.min(orig_w - 1);
        let crop_y = by.min(orig_h - 1);
        let crop_w = bw.min(orig_w - crop_x);
        let crop_h = bh.min(orig_h - crop_y);

        if crop_w == 0 || crop_h == 0 {
            continue;
        }

        // 裁剪出当前文本块图像
        let cropped = img.crop_imm(crop_x, crop_y, crop_w, crop_h);
        
        // 识别输入图像预处理 (固定高度为 48, 宽度根据比例自适应)
        let rec_h = 48;
        let rec_w = (crop_w as f32 * (rec_h as f32 / crop_h as f32)) as u32;
        let rec_input = preprocess_rec(&cropped, rec_w, rec_h);

        // 运行文本识别模型
        let rec_input_value = ort::value::Value::from_array(rec_input.clone())?;
        let rec_outputs = rec_session.run(ort::inputs![rec_input_value])?;
        let rec_tensor = rec_outputs[0].try_extract_array::<f32>()?;

        // 8. CTC 解码得到识别文本
        let text = decode_ctc(&rec_tensor, &dict);
        println!("  👉 [框 {}] 坐标:({},{},{},{}) -> 识别结果: \"{}\"", i + 1, bx, by, bw, bh, text);
    }

    println!("✨ OCR 任务处理完成！");
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
