import os
import tempfile
from contextlib import contextmanager
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from funasr import AutoModel
import uvicorn
import logging

# --- 配置日志 ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 全局模型变量 ---
MODEL_DIR = "FunAudioLLM/Fun-ASR-Nano-2512"
MODEL_INSTANCE = None

def load_model():
    """加载模型的函数，便于错误处理"""
    global MODEL_INSTANCE
    try:
        logger.info(f"正在加载模型: {MODEL_DIR}...")
        MODEL_INSTANCE = AutoModel(
            model=MODEL_DIR,
            trust_remote_code=True,
            remote_code="./model.py", # 请确保此文件存在
            device="cpu", # 可根据环境改为 "cuda"
            hub="ms"
        )
        logger.info("模型加载完成。")
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        raise

@contextmanager
def save_upload_file_to_temp(upload_file: UploadFile):
    """上下文管理器：安全地将上传文件保存到临时文件"""
    # 获取原始文件名，并处理可能为 None 的情况
    original_filename: Optional[str] = upload_file.filename
    if not original_filename:
        # 如果没有文件名，则使用一个通用的后缀
        suffix = ".tmp"
    else:
        # 安全地获取后缀
        parts = original_filename.rsplit('.', 1)
        suffix = f".{parts[1]}" if len(parts) > 1 else ""

    # 使用 tempfile 创建临时文件
    temp_fd, temp_path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(temp_fd, 'wb') as temp_file:
            # 将上传文件的内容读取并写入临时文件
            content = upload_file.file.read()
            temp_file.write(content)
        yield temp_path
    finally:
        # 确保临时文件被清理
        if os.path.exists(temp_path):
            os.unlink(temp_path)

# --- FastAPI 应用 ---
app = FastAPI(
    title="FunASR OpenAI Compatible API",
    description="A local API compatible with OpenAI's audio transcription endpoint.",
    version="1.0.0"
)

# 将启动事件的注册放在 app 定义之后
@app.on_event('startup')
def startup_event():
    load_model()

@app.post("/v1/audio/transcriptions", 
          summary="Create Transcription",
          description="Transcribes audio into the input language.")
async def create_transcription(
    file: UploadFile = File(..., description="The audio file to transcribe (WAV, MP3, etc.)"),
    model_name: str = Form(None, alias="model", description="ID of the model to use. Ignored for local models."),
    language: str = Form(None, description="The language of the input audio. Supplying the input language provides better accuracy."),
    prompt: str = Form(None, description="An optional text to guide the model's style or continue a previous segment."),
    response_format: str = Form("json", description="The format of the transcript output."),
    temperature: float = Form(0, description="The sampling temperature, between 0 and 1. Higher values make output more random.")
):
    """
    OpenAI 兼容的语音转文字接口
    对应 OpenAI API: POST /v1/audio/transcriptions
    """
    if not MODEL_INSTANCE:
        logger.error("模型未加载，无法处理请求。")
        return JSONResponse(status_code=500, content={"error": "Server model is not loaded properly."})

    try:
        # 使用上下文管理器安全保存文件
        with save_upload_file_to_temp(file) as temp_file_path:
            # --- 准备推理参数 ---
            generate_kwargs = {
                "input": [temp_file_path],
                "cache": {},
                "batch_size": 1,
                "itn": True, # 默认开启逆文本标准化
            }

            # 处理语言参数
            if language:
                # OpenAI 使用 'zh', 'en', 'ja' 等，FunASR 使用 '中文', '英文', '日文'
                lang_map = {"zh": "中文", "en": "英文", "ja": "日文"}
                generate_kwargs["language"] = lang_map.get(language, language)
            
            # 处理 prompt (示例：作为热词)
            if prompt:
                # 注意：热词功能的效果取决于模型本身的支持情况
                generate_kwargs["hotwords"] = [prompt]

            # --- 调用 FunASR 推理 ---
            logger.info(f"开始处理文件: {file.filename or 'unknown'}")
            res = MODEL_INSTANCE.generate(**generate_kwargs)
            logger.info(f"处理完成: {file.filename or 'unknown'}")

        # 提取结果
        if not res or len(res) == 0 or "text" not in res[0]:
             logger.error(f"模型返回结果格式异常: {res}")
             return JSONResponse(status_code=500, content={"error": "Model returned invalid result."})
             
        text = res[0]["text"]

        # --- 返回 OpenAI 标准格式 ---
        # 这里可以根据 response_format 参数进一步扩展返回格式
        return JSONResponse(content={"text": text})

    except Exception as e:
        logger.error(f"处理请求时发生错误: {e}")
        # 注意：由于使用了上下文管理器，临时文件已在退出时被清理
        return JSONResponse(status_code=500, content={"error": f"Internal Server Error: {str(e)}"})

@app.get("/health", summary="Health Check", description="Check if the service and model are running.")
async def health():
    status = "running" if MODEL_INSTANCE else "model not loaded"
    return JSONResponse(content={"status": status, "model": MODEL_DIR})

if __name__ == "__main__":
    # 启动服务，监听 0.0.0.0:8000
    uvicorn.run(app, host="0.0.0.0", port=8000)