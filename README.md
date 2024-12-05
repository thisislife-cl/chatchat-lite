# Chatchat-lite

## 运行环境
Python >= 3.9
建议使用 3.10

可参考如下命令进行环境创建
```commandline
conda create -n chatchat-lite python=3.10 -y
conda activate chatchat-lite
```


## 安装依赖
```commandline
pip install -r requirements.txt
```

## 启动本地模型
当前项目仅支持接入 Ollama 模型
请前往 [Ollama官网](https://ollama.com/download) 下载最新版 Ollama， 安装完成后再命令行中执行以下命令：
```commandline
ollama run qwen2.5
```

## 运行项目
```commandline
streamlit run st_main.py
```
