import os
import glob
import datetime

WORK_DIR = r"C:\Users\周正\Desktop\33550336\Ayxi\Ayin\eyes"
TODAY = datetime.datetime.now().strftime("%Y年%m月%d日")
LOG_FILE = os.path.join(WORK_DIR, f"研究日志_{TODAY}.md")

def get_recent_files():
    patterns = ["*.py", "*.txt", "*.md"]
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(WORK_DIR, p)))
    files = [f for f in files if os.path.basename(f) not in ["auto_daily_log.py"]]
    files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return files[:10]

def analyze_activity():
    files = get_recent_files()
    activities = []
    
    for f in files:
        name = os.path.basename(f)
        mtime = datetime.datetime.fromtimestamp(os.path.getmtime(f))
        
        if name == "debug_log.txt":
            activities.append(f"调试日志更新于 {mtime.strftime('%H:%M')}")
        elif "hand" in name.lower() and name.endswith(".py"):
            activities.append(f"手势控制模块 {name}")
        elif name.endswith(".md"):
            activities.append(f"文档 {name}")
        else:
            activities.append(f"代码 {name}")
    
    return activities[:5]

def generate_log():
    log_content = f"""# 研究日志

**日期**：{TODAY}  
**研究者**：霍恩海姆  
**状态**：疲惫的研究员

---

## 今日活动

{datetime.datetime.now().strftime("%H:%M")} 自动生成

"""
    
    activities = analyze_activity()
    if activities:
        log_content += "### 主要工作\n\n"
        for a in activities:
            log_content += f"- {a}\n"
        log_content += "\n"
    else:
        log_content += "### 主要工作\n\n- 无活动记录\n\n"
    
    log_content += """---

*此日志由自动脚本生成*

"""
    
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(log_content)
    
    print(f"已生成: {LOG_FILE}")

if __name__ == "__main__":
    generate_log()