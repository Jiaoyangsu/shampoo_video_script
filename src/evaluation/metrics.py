import json
import re
import sys

PROMPTS = [
    {
        "short": "海飞丝去屑",
        "keywords": ["去屑", "头屑", "海飞丝"],
        "full": "帮我生成一个海飞丝洗发水的带货短视频分镜脚本，核心卖点是去屑，需要展示头皮清洁的痛点。",
    },
    {
        "short": "婴儿无泪",
        "keywords": ["无泪", "温和", "婴儿", "宝宝"],
        "full": "设计一个婴儿洗发水视频脚本，强调无泪配方和温和，妈妈放心。",
    },
    {
        "short": "男士控油蓬松",
        "keywords": ["控油", "蓬松", "油性", "上班族"],
        "full": "做一个控油蓬松男士洗发水的带货脚本，目标用户是油性头皮、发量扁塌的上班族，预算有限，需要突出性价比和即时效果。",
    },
]

UNKNOWN_BRANDS = ["贝德美", "BODORME"]
EXPECTED_BRANDS = ["海飞丝", "头屑净"]


def strip_fence(text):
    m = re.search(r"```json\s*(.*?)\s*```", text, re.S)
    return m.group(1) if m else text


def parse_time(t):
    m = re.match(r"(\d{1,2}):(\d{2})(?:-(\d{1,2}):(\d{2}))?", t or "")
    if not m:
        return None
    start = int(m.group(1)) * 60 + int(m.group(2))
    if m.group(3):
        end = int(m.group(3)) * 60 + int(m.group(4))
        return (start, end)
    return (start, None)


def score_log(path):
    content = open(path, encoding="utf-8").read()
    blocks = content.split("=" * 70)
    results = []
    for b in blocks:
        if "PROMPT:" not in b:
            continue
        full_prompt = re.search(r"PROMPT: (.+)", b).group(1).strip()
        raw_m = re.search(r"RAW:\n(.*?)(?=\n-{70})", b, re.S)
        raw = raw_m.group(1) if raw_m else ""
        parsed, js = None, raw
        try:
            parsed = json.loads(strip_fence(raw))
        except Exception:
            pass
        shots = parsed if isinstance(parsed, list) else []
        if not shots:
            objs = re.findall(r"\{[^{}]*\}", strip_fence(raw))
            for o in objs:
                try:
                    shots.append(json.loads(o))
                except Exception:
                    pass
        body = js
        n = len(shots)
        valid = bool(parsed)
        shot_types = set()
        monotonic = True
        last_end = -1
        gaps = 0
        overlaps = 0
        for s in shots:
            t = s.get("time_axis", "")
            shot_types.add(s.get("shot_type", ""))
            st = parse_time(t)
            if st:
                start, end = st
                if end is None:
                    end = start
                if start < last_end:
                    overlaps += 1
                if start > last_end and last_end >= 0:
                    gaps += 1
                last_end = end if end > last_end else last_end
            if not (s.get("time_axis") and s.get("shot_type") and s.get("visual_description") and s.get("voiceover") and s.get("music")):
                monotonic = False
        keywords = next(
            (p["keywords"] for p in PROMPTS if p["full"].strip() == full_prompt.strip()), []
        )
        if not keywords:
            keywords = next(
                (p["keywords"] for p in PROMPTS if p["short"] in full_prompt), []
            )
        keyword_hits = sum(1 for k in keywords if k.lower() in body.lower())
        unknown_brand = next((b0 for b0 in UNKNOWN_BRANDS if b0 in body), None)
        expected = next(
            (b0 for b0 in EXPECTED_BRANDS if b0 in body), None
        ) if "海飞丝" in full_prompt else None
        empty_vo = body.count('"voiceover": ""') + body.count('"voiceover": "无"')
        results.append({
            "prompt": full_prompt[:22],
            "n_shots": n,
            "json_valid": valid,
            "keyword_hits": f"{keyword_hits}/{len(keywords)}",
            "distinct_types": len(shot_types),
            "monotonic_time": monotonic,
            "gaps": gaps,
            "overlaps": overlaps,
            "empty_voiceover": empty_vo,
            "unknown_brand": unknown_brand,
            "expected_brand": expected,
        })
    return results


def main():
    tag = sys.argv[1]
    path = sys.argv[2]
    print(f"===== OBJECTIVE METRICS: {tag} =====")
    print(f"{'prompt':<22} {'#shots':>6} {'json':>4} {'kw':>7} {'types':>5} {'mono':>5} {'gaps':>4} {'overlap':>7} {'emptyVO':>7} {'brand':<10}")
    for r in score_log(path):
        print(
            f"{r['prompt']:<22} {r['n_shots']:>6} {str(r['json_valid']):>4} {r['keyword_hits']:>7} "
            f"{r['distinct_types']:>5} {str(r['monotonic_time']):>5} {r['gaps']:>4} {r['overlaps']:>7} "
            f"{r['empty_voiceover']:>7} {r['unknown_brand'] or r['expected_brand'] or '-':<10}"
        )
    print("METRICS_DONE")


if __name__ == "__main__":
    main()
