import requests
import gradio as gr
import sounddevice as sd
import numpy as np
import os
import tempfile
import json
import time
from datetime import datetime
import scipy.io.wavfile as wav
import logging
import warnings
from urllib3.exceptions import InsecureRequestWarning
import wave
import contextlib
import subprocess
import sys
import zipfile
import urllib.request
import shutil
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# DeepSeek API 配置
DEEPSEEK_API_KEY = "sk-e07024fd50f74c058a961512fcdabfa7"
CHAT_API_URL = "https://api.deepseek.com/v1/chat/completions"

# 全局变量
meeting_data = {
    "transcript": "",
    "start_time": "",
    "speakers": [],
    "key_points": [],
    "action_items": []
}

# Vosk模型配置
VOSK_MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-small-cn-0.22.zip"
VOSK_MODEL_ZIP = "vosk-model-small-cn-0.22.zip"
VOSK_MODEL_DIR = "vosk-model-small-cn-0.22"

# 更新为您的 FFmpeg 路径
FFMPEG_BIN_DIR = r"C:\Users\liuziyu\ffmpeg-2025-07-01-git-11d1b71c31-full_build\bin"
FFMPEG_PATH = os.path.join(FFMPEG_BIN_DIR, "ffmpeg.exe")
FFPROBE_PATH = os.path.join(FFMPEG_BIN_DIR, "ffprobe.exe")

def check_ffmpeg_installed():
    """检查系统是否安装了FFmpeg"""
    # 检查直接指定的路径是否存在
    if os.path.exists(FFMPEG_PATH) and os.path.exists(FFPROBE_PATH):
        logger.info(f"找到FFmpeg可执行文件: {FFMPEG_PATH}")
        return True
    
    logger.warning(f"在指定路径未找到FFmpeg: {FFMPEG_PATH}")
    return False

# 新增：转换音频格式为WAV
def convert_to_wav(input_path):
    """将任意音频格式转换为WAV格式"""
    if not input_path:
        return None
    
    # 已经是WAV格式，直接返回
    if input_path.lower().endswith('.wav'):
        return input_path
    
    # 创建临时WAV文件
    wav_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    
    try:
        # 使用FFmpeg进行转换
        cmd = [
            FFMPEG_PATH,        # 使用直接指定的路径
            '-i', input_path,   # 输入文件
            '-ac', '1',         # 单声道
            '-ar', '16000',     # 16kHz采样率
            '-y',               # 覆盖输出文件
            wav_path            # 输出文件
        ]
        
        # 在Windows上隐藏控制台窗口
        startupinfo = None
        if sys.platform.startswith('win'):
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = 0
        
        result = subprocess.run(cmd, 
                              stdout=subprocess.PIPE, 
                              stderr=subprocess.PIPE,
                              text=True,
                              startupinfo=startupinfo)
        
        if result.returncode != 0:
            logger.error(f"音频转换失败: {result.stderr}")
            return None
        
        logger.info(f"音频转换成功: {input_path} -> {wav_path}")
        return wav_path
    except Exception as e:
        logger.error(f"音频转换异常: {str(e)}")
        return None

def fix_model_directory():
    """修复模型目录结构"""
    # 检查是否存在嵌套目录
    nested_dir = os.path.join(VOSK_MODEL_DIR, "vosk-model-small-cn-0.22")
    
    if os.path.exists(nested_dir):
        logger.info(f"检测到嵌套目录: {nested_dir}")
        logger.info("正在修复目录结构...")
        
        # 移动所有文件到父目录
        for item in os.listdir(nested_dir):
            src = os.path.join(nested_dir, item)
            dst = os.path.join(VOSK_MODEL_DIR, item)
            
            # 如果目标已存在，先删除
            if os.path.exists(dst):
                if os.path.isdir(dst):
                    shutil.rmtree(dst)
                else:
                    os.remove(dst)
            
            shutil.move(src, dst)
        
        # 删除空的嵌套目录
        shutil.rmtree(nested_dir)
        logger.info("目录结构修复完成")
    else:
        logger.info("目录结构正常，无需修复")

def download_and_extract_model():
    """下载并解压Vosk中文模型"""
    # 检查模型目录是否存在
    if os.path.exists(VOSK_MODEL_DIR):
        logger.info(f"Vosk模型已存在: {VOSK_MODEL_DIR}")
        fix_model_directory()  # 确保目录结构正确
        return True
    
    logger.info(f"开始下载Vosk中文模型: {VOSK_MODEL_URL}")
    
    try:
        # 下载模型
        urllib.request.urlretrieve(VOSK_MODEL_URL, VOSK_MODEL_ZIP)
        logger.info(f"模型下载完成: {VOSK_MODEL_ZIP}")
        
        # 解压模型
        with zipfile.ZipFile(VOSK_MODEL_ZIP, 'r') as zip_ref:
            zip_ref.extractall(".")
        logger.info(f"模型解压完成: {VOSK_MODEL_DIR}")
        
        # 修复目录结构
        fix_model_directory()
        
        # 删除ZIP文件
        os.remove(VOSK_MODEL_ZIP)
        logger.info(f"已删除临时文件: {VOSK_MODEL_ZIP}")
        
        return True
    except Exception as e:
        logger.error(f"模型下载或解压失败: {str(e)}")
        return False

def install_vosk():
    """安装Vosk库"""
    try:
        import vosk
        logger.info("Vosk库已安装")
        return True
    except ImportError:
        logger.info("正在安装Vosk库...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "vosk"])
            logger.info("Vosk库安装成功")
            return True
        except Exception as e:
            logger.error(f"Vosk库安装失败: {str(e)}")
            return False

def create_retry_session():
    """创建带重试机制的请求会话"""
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    return session

def transcribe_audio_local(audio_path):
    """使用本地Vosk引擎进行语音识别"""
    try:
        import vosk
    except ImportError:
        return "错误: Vosk库未安装，请检查安装"
    
    # 确保模型存在
    if not os.path.exists(VOSK_MODEL_DIR):
        logger.error(f"Vosk模型目录不存在: {VOSK_MODEL_DIR}")
        return "错误: Vosk模型未找到"
    
    try:
        # 打印模型目录内容以调试
        logger.info(f"模型目录内容: {os.listdir(VOSK_MODEL_DIR)}")
        
        # 检查关键目录是否存在
        required_dirs = ["am", "conf", "graph"]
        for dir_name in required_dirs:
            dir_path = os.path.join(VOSK_MODEL_DIR, dir_name)
            if not os.path.exists(dir_path):
                logger.error(f"关键目录缺失: {dir_path}")
                return f"错误: 模型不完整，缺失 {dir_name} 目录"
        
        # 加载模型
        model = vosk.Model(VOSK_MODEL_DIR)
        logger.info("Vosk模型加载成功")
        
        # 读取音频文件
        wf = wave.open(audio_path, "rb")
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
            logger.warning("音频格式不符合要求")
            return "错误: 音频格式必须是16位PCM单声道WAV"
        
        # 创建识别器
        rec = vosk.KaldiRecognizer(model, wf.getframerate())
        rec.SetWords(True)
        
        # 处理音频
        results = []
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                results.append(result.get("text", ""))
        
        # 获取最终结果
        final_result = json.loads(rec.FinalResult())
        results.append(final_result.get("text", ""))
        
        return " ".join(results).strip()
    
    except Exception as e:
        logger.error(f"本地语音识别失败: {str(e)}")
        return f"错误: 本地语音识别失败 - {str(e)}"

def transcribe_audio(audio_path):
    """音频转录主函数"""
    # 检查文件大小
    file_size = os.path.getsize(audio_path)
    if file_size > 25 * 1024 * 1024:  # 25MB
        return "错误: 音频文件过大 (超过25MB)，请压缩或使用较短录音"
    
    # 获取音频时长
    try:
        with contextlib.closing(wave.open(audio_path, 'r')) as f:
            duration = f.getnframes() / f.getframerate()
    except:
        try:
            # 尝试使用FFprobe获取时长
            cmd = [
                FFPROBE_PATH,  # 使用直接指定的路径
                '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                audio_path
            ]
            
            # 在Windows上隐藏控制台窗口
            startupinfo = None
            if sys.platform.startswith('win'):
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = 0
            
            result = subprocess.run(cmd, 
                                  stdout=subprocess.PIPE, 
                                  stderr=subprocess.PIPE,
                                  text=True,
                                  startupinfo=startupinfo)
            
            if result.returncode == 0:
                duration = float(result.stdout.strip())
            else:
                duration = 0
        except:
            duration = 0
    
    if duration > 600:  # 10分钟
        return "错误: 音频过长 (超过10分钟)，请分段处理"
    
    # 使用本地引擎进行转录
    return transcribe_audio_local(audio_path)

def deepseek_chat(prompt, system_prompt="你是一个专业的AI助手", model="deepseek-chat"):
    """调用DeepSeek文本生成API"""
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 2000
    }
    
    try:
        session = create_retry_session()
        response = session.post(
            CHAT_API_URL, 
            headers=headers, 
            json=payload, 
            timeout=60,
            verify=True
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            error_msg = f"文本生成失败: {response.status_code} - {response.text[:200]}"
            logger.error(error_msg)
            return error_msg
    
    except requests.exceptions.RequestException as e:
        error_msg = f"请求异常: {str(e)}"
        logger.error(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"未知错误: {str(e)}"
        logger.error(error_msg)
        return error_msg

def record_audio(duration=60, sample_rate=16000):
    """录制音频"""
    logger.info(f"开始录音，时长: {duration}秒...")
    
    # 使用更高效的设置
    channels = 1  # 单声道
    dtype = np.int16  # 16位整型
    
    # 计算采样点数
    samples = int(duration * sample_rate)
    
    # 录制音频
    audio = sd.rec(samples, samplerate=sample_rate, channels=channels, dtype=dtype)
    sd.wait()
    logger.info("录音结束")
    
    # 保存为临时文件
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    wav.write(temp_file.name, sample_rate, audio)
    
    # 检查文件大小
    file_size = os.path.getsize(temp_file.name)
    logger.info(f"录音文件大小: {file_size/1024:.2f}KB")
    
    return temp_file.name

def process_meeting(audio_path=None, text_input=None):
    """处理会议录音或文本"""
    global meeting_data
    
    meeting_data = {
        "transcript": "",
        "start_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "speakers": [],
        "key_points": [],
        "action_items": []
    }
    
    if audio_path:
        # 新增：转换音频格式为WAV
        original_path = audio_path
        converted_path = None
        
        # 检查是否需要转换
        if not audio_path.lower().endswith('.wav'):
            logger.info(f"转换音频格式: {audio_path}")
            converted_path = convert_to_wav(audio_path)
            
            if converted_path:
                audio_path = converted_path
                logger.info(f"使用转换后的文件: {audio_path}")
            else:
                return "错误: 音频格式转换失败，请确保已安装FFmpeg", "无法生成摘要"
        
        # 处理音频
        # 检查文件大小
        file_size = os.path.getsize(audio_path)
        if file_size > 25 * 1024 * 1024:  # 25MB
            return "错误: 音频文件过大 (超过25MB)", "无法生成摘要"
        
        transcript = transcribe_audio(audio_path)
        
        # 删除转换后的临时文件
        if converted_path and os.path.exists(converted_path):
            try:
                os.remove(converted_path)
                logger.info(f"已删除临时文件: {converted_path}")
            except Exception as e:
                logger.error(f"删除临时文件失败: {str(e)}")
        
        if "失败" in transcript or "错误" in transcript:
            return transcript, "无法生成摘要"
        
        # 如果转写成功但内容为空
        if not transcript.strip():
            return "语音识别成功但返回空内容", "请检查音频质量或重试"
        
        meeting_data["transcript"] = transcript
    elif text_input:
        # 直接使用输入的文本
        meeting_data["transcript"] = text_input
        transcript = text_input
    else:
        return "错误: 未提供音频输入或文本", "请上传音频或输入文本"
    
    # 生成会议摘要
    summary_prompt = f"""请根据以下会议记录生成结构化摘要：
    
    ## 输出格式要求：
    - 主要参会人员: [列出主要发言人]
    - 核心议题: [列出3-5个关键议题]
    - 重要结论: [列出重要决定和结论]
    - 待办事项: 
      - [任务1] 负责人: [姓名] 截止时间: [日期]
      - [任务2] 负责人: [姓名] 截止时间: [日期]
    
    会议记录内容：
    {meeting_data["transcript"][:3000]} [仅显示前3000字符]
    """
    
    summary = deepseek_chat(summary_prompt, "你是一个专业的会议记录助手")
    meeting_data["summary"] = summary
    
    return transcript, summary

def generate_news_report(style="正式报道"):
    """生成会议新闻报道"""
    if not meeting_data.get("summary"):
        return "错误: 请先处理会议录音或文本"
    
    prompt = f"""请根据以下会议摘要生成一篇{style}风格的新闻报道：
    
    {meeting_data.get('summary', '')}
    
    报道要求：
    1. 标题醒目
    2. 包含会议基本信息（时间、地点、参会人员）
    3. 突出会议成果和重要决策
    4. 使用{style}风格撰写
    """
    
    styles = {
        "正式报道": "专业、严谨的新闻报道风格",
        "简报摘要": "简洁明了的要点式总结",
        "社交媒体": "生动活泼的社交媒体风格，使用表情符号和话题标签"
    }
    
    return deepseek_chat(prompt, f"你是一名记者，擅长写{styles[style]}的会议报道")

def check_dependencies():
    """检查并安装必要的依赖"""
    logger.info("检查依赖项...")
    
    # 检查FFmpeg
    ffmpeg_installed = check_ffmpeg_installed()
    if not ffmpeg_installed:
        logger.error(f"未找到FFmpeg: {FFMPEG_PATH}")
        logger.error("请确保FFmpeg已安装并更新代码中的路径")
        return False
    
    # 检查并安装vosk
    if not install_vosk():
        return False
    
    # 检查并下载模型
    if not download_and_extract_model():
        return False
    
    return True

# 创建Gradio界面
with gr.Blocks(theme=gr.themes.Soft(), title="DeepSeek会议助手") as app:
    gr.Markdown("# 🎙️ DeepSeek会议记录助手")
    gr.Markdown("使用国产DeepSeek AI技术，自动生成会议记录和新闻报道")
    
    with gr.Tab("上传会议录音"):
        gr.Markdown("### 上传会议录音文件 (支持WAV/M4A/MP3格式)")
        gr.Markdown("**注意**: 文件大小需小于25MB，建议录制1-5分钟音频")
        # 修改：支持多种格式
        audio_input = gr.File(
            file_types=["audio"], 
            label="上传录音文件 (支持WAV/M4A/MP3)",
            type="filepath"
        )
        upload_btn = gr.Button("处理会议录音", variant="primary")
    
    with gr.Tab("实时会议记录"):
        with gr.Row():
            duration = gr.Slider(1, 180, value=60, label="录音时长(秒)",  # 最大3分钟
                                info="建议不超过3分钟")
            record_btn = gr.Button("开始录音", variant="primary")
        record_status = gr.Textbox(label="录音状态", interactive=False, value="就绪")
        record_output = gr.Audio(label="录音预览", interactive=False)
    
    with gr.Tab("输入会议文本"):
        gr.Markdown("### 直接输入会议文本")
        # 增加文本输入框的高度
        text_input = gr.Textbox(label="会议文本", lines=15, placeholder="在此粘贴会议记录文本...")
        text_btn = gr.Button("处理文本", variant="primary")
    
    with gr.Tab("会议结果"):
        # 将转录文本放入折叠面板
        with gr.Accordion("会议转录文本（点击展开）", open=False):
            transcript = gr.Textbox(interactive=False, show_label=False, 
                                  lines=10)
        
        # 优化摘要框 - 增加高度
        summary = gr.Textbox(label="会议摘要", interactive=False, 
                           lines=8)
        
        with gr.Row():
            style = gr.Radio(
                choices=["正式报道", "简报摘要", "社交媒体"],
                value="正式报道",
                label="新闻风格"
            )
            news_btn = gr.Button("生成新闻报道", variant="primary")
        
        # 增加新闻报告框的高度
        news_report = gr.Textbox(label="会议新闻报道", interactive=False, 
                                lines=12)
    
    # 状态信息
    status_info = gr.Textbox(label="系统状态", interactive=False)
    
    # 添加自定义CSS样式 - 解决滚动和换行问题
    app.css = """
    textarea {
        max-height: 500px !important;
        overflow-y: auto !important;
    }
    .wrap {
        white-space: pre-wrap !important;
    }
    .scrollable {
        max-height: 500px;
        overflow-y: auto;
    }
    """
    
    # 事件处理
    def handle_upload(audio_path):
        if not audio_path:
            return "请上传音频文件", "", "错误: 未提供音频输入", "就绪"
        
        # 检查文件大小
        file_size = os.path.getsize(audio_path) if audio_path else 0
        if file_size > 25 * 1024 * 1024:  # 25MB
            return audio_path, "错误: 文件过大 (超过25MB)", "请压缩文件或录制较短音频", "错误: 文件过大"
        
        status_info.value = "开始处理音频..."
        try:
            transcript_text, summary_text = process_meeting(audio_path=audio_path)
            status_info.value = "处理完成"
            return audio_path, transcript_text, summary_text, "就绪"
        except Exception as e:
            logger.error(f"处理失败: {str(e)}")
            return audio_path, f"处理失败: {str(e)}", "请重试", f"错误: {str(e)}"
    
    upload_btn.click(
        fn=handle_upload,
        inputs=audio_input,
        outputs=[audio_input, transcript, summary, status_info]
    )
    
    def handle_text_input(text_input):
        if not text_input.strip():
            return "", "错误: 文本内容为空", "请输入会议文本", "错误: 文本为空"
        
        status_info.value = "开始处理文本..."
        try:
            transcript_text, summary_text = process_meeting(text_input=text_input)
            status_info.value = "处理完成"
            return text_input, transcript_text, summary_text, "就绪"
        except Exception as e:
            logger.error(f"处理失败: {str(e)}")
            return text_input, f"处理失败: {str(e)}", "请重试", f"错误: {str(e)}"
    
    text_btn.click(
        fn=handle_text_input,
        inputs=text_input,
        outputs=[text_input, transcript, summary, status_info]
    )
    
    def record_and_process(duration):
        """录音并处理"""
        if duration > 180:  # 3分钟
            return None, "错误: 录音时长不能超过3分钟", "请设置较短时长", "错误: 超时"
        
        record_status.value = "录音中..."
        status_info.value = "开始录音..."
        try:
            audio_file = record_audio(duration)
            record_status.value = "录音完成"
            status_info.value = "开始处理录音..."
            transcript_text, summary_text = process_meeting(audio_path=audio_file)
            status_info.value = "处理完成"
            return audio_file, transcript_text, summary_text, "就绪"
        except Exception as e:
            logger.error(f"录音处理失败: {str(e)}")
            return None, f"录音失败: {str(e)}", "请重试", f"错误: {str(e)}"
    
    record_btn.click(
        fn=record_and_process,
        inputs=duration,
        outputs=[record_output, transcript, summary, status_info]
    ).then(
        lambda: "就绪",
        outputs=record_status
    )
    
    news_btn.click(
        fn=generate_news_report,
        inputs=style,
        outputs=news_report
    )

if __name__ == "__main__":
    # 禁用SSL警告
    warnings.filterwarnings("ignore", category=InsecureRequestWarning)
    
    # 打印配置信息
    print(f"DeepSeek文本API端点: {CHAT_API_URL}")
    print(f"FFmpeg路径: {FFMPEG_PATH}")
    print(f"FFprobe路径: {FFPROBE_PATH}")
    
    # 检查并安装依赖
    print("正在检查依赖项...")
    if not check_dependencies():
        print("⚠️ 依赖项检查失败，请确保:")
        print(f"1. FFmpeg已安装在: {FFMPEG_PATH}")
        print("2. Vosk模型已正确下载和解压")
        print("3. Vosk库已安装 (pip install vosk)")
    else:
        print("✓ 所有依赖项已准备就绪")
    
    # 启动应用
    app.launch(server_name="0.0.0.0", server_port=51551)
   