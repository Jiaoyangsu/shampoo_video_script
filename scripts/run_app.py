#!/usr/bin/env python3
"""
启动 Web 界面
"""
import os
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.app.chat_ui import demo

if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))
    demo.queue().launch(server_port=9999)
