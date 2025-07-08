import requests
import gradio as gr
import sounddevice as sd
import numpy as np
import os
import tempfile
import json
import time
from datetime import datetime

# DeepSeek API 配置
DEEPSEEK_API_KEY = "your_api_key_here"  # 替换为您的API密钥
SPEECH_API_URL = "https://api.deepseek.com/v1/audio/transcriptions"
CHAT_API_URL = "https://api.deepseek.com/v1/chat/completions"

# 全局变量
is_recording = False
meeting_data = {
    "transcript": "",
    "start_time": "",
    "speakers": [],
    "key_points": [],
    "action_items": []
}

def transcribe_audio(audio_path):
    """使用DeepSeek语音识别API转写音频"""
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "multipart/form-data"
    }
    
    with open(audio_path, "rb") as audio_file:
        files = {"file": (os.path.basename(audio_path), audio_file, "audio/wav")}
        data = {"model": "deepseek-vl"}
        
        response = requests.post(SPEECH_API_URL, headers=headers, files=files, data=data)
    
    if response.status_code == 200:
        result = response.json()
        return result.get("text", "")
    else:
        print(f"语音识别失败: {response.status_code} - {response.text}")
        return ""

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
    
    response = requests.post(CHAT_API_URL, headers=headers, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        return result["choices"][0]["message"]["content"]
    else:
        print(f"文本生成失败: {response.status_code} - {response.text}")
        return ""

def record_audio(duration=60, sample_rate=16000):
    """录制音频"""
    print(f"开始录音，时长: {duration}秒...")
    audio = sd.rec(int(duration * sample_rate), 
                   samplerate=sample_rate, channels=1)
    sd.wait()
    print("录音结束")
    
    # 保存临时文件
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    audio.tofile(temp_file.name)
    return temp_file.name

def process_meeting(audio_path=None, live_audio=None):
    """处理会议录音"""
    global meeting_data
    
    # 重置会议数据
    meeting_data = {
        "transcript": "",
        "start_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "speakers": [],
        "key_points": [],
        "action_items": []
    }
    
    # 获取音频内容
    if audio_path:
        # 处理上传的音频文件
        transcript = transcribe_audio(audio_path)
    elif live_audio:
        # 处理实时录音
        with open(live_audio, "wb") as f:
            f.write(live_audio)
        transcript = transcribe_audio(live_audio)
    else:
        return "错误: 未提供音频输入", ""
    
    meeting_data["transcript"] = transcript
    
    # 生成会议摘要
    summary_prompt = f"""
    请根据以下会议记录生成结构化摘要：
    
    ## 输出格式要求：
    - 主要参会人员: [列出主要发言人]
    - 核心议题: [列出3-5个关键议题]
    - 重要结论: [列出重要决定和结论]
    - 待办事项: 
      - [任务1] 负责人: [姓名] 截止时间: [日期]
      - [任务2] 负责人: [姓名] 截止时间: [日期]
    
    会议记录内容：
    {transcript[:5000]}  # 限制长度
    """
    
    summary = deepseek_chat(summary_prompt, "你是一个专业的会议记录助手")
    return transcript, summary

def generate_news_report(style="正式报道"):
    """生成会议新闻报道"""
    prompt = f"""
    请根据以下会议摘要生成一篇{style}风格的新闻报道：
    
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

# 创建Gradio界面
with gr.Blocks(theme=gr.themes.Soft(), title="DeepSeek会议助手") as app:
    gr.Markdown("# 🎙️ DeepSeek会议记录助手")
    gr.Markdown("使用国产DeepSeek AI技术，自动生成会议记录和新闻报道")
    
    with gr.Tab("上传会议录音"):
        audio_input = gr.Audio(source="upload", type="filepath", label="上传录音文件")
        upload_btn = gr.Button("处理会议录音", variant="primary")
    
    with gr.Tab("实时会议记录"):
        with gr.Row():
            duration = gr.Slider(1, 600, value=60, label="录音时长(秒)")
            record_btn = gr.Button("开始录音", variant="primary")
        record_output = gr.Audio(label="录音预览", interactive=False)
    
    with gr.Tab("会议结果"):
        transcript = gr.Textbox(label="会议转录文本", lines=10)
        summary = gr.Textbox(label="会议摘要", lines=10)
        
        with gr.Row():
            style = gr.Radio(
                choices=["正式报道", "简报摘要", "社交媒体"],
                value="正式报道",
                label="新闻风格"
            )
            news_btn = gr.Button("生成新闻报道", variant="primary")
        news_report = gr.Textbox(label="会议新闻报道", lines=12)
    
    # 事件处理
    upload_btn.click(
        fn=process_meeting,
        inputs=audio_input,
        outputs=[transcript, summary]
    )
    
    record_btn.click(
        fn=record_audio,
        inputs=duration,
        outputs=record_output
    )
    
    record_output.change(
        fn=lambda x: process_meeting(live_audio=x),
        inputs=record_output,
        outputs=[transcript, summary]
    )
    
    news_btn.click(
        fn=generate_news_report,
        inputs=style,
        outputs=news_report
    )

if __name__ == "__main__":
    app.launch()