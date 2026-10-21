import json


def make_script(n=10, chars=8, music="轻快背景音乐"):
    shots = []
    for i in range(n):
        start = i * 3
        end = start + 3
        shots.append(
            {
                "time_axis": "%02d:%02d-%02d:%02d" % (start // 60, start % 60, end // 60, end % 60),
                "shot_type": "痛点展示" if i == 0 else "产品展示",
                "visual_description": "镜头%d展示产品使用场景和人物动作。" % (i + 1),
                "voiceover": "这是第%d个镜头的口播" % (i + 1) + "好" * max(0, chars - 8),
                "music": music,
            }
        )
    return shots


def make_row(n=10, input_text="请为海飞丝去屑洗发水写一条30秒带货脚本。"):
    return {"input": input_text, "output": make_script(n)}


def dump(value):
    return json.dumps(value, ensure_ascii=False)
