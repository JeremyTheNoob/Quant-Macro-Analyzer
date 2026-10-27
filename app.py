import os
import sys
import subprocess
import threading
from flask import Flask, render_template, jsonify, send_from_directory

print("=" * 60, flush=True)
print("Python:", sys.executable, flush=True)
print("目录:", os.getcwd(), flush=True)
print("templates/index.html 存在:",
      os.path.isfile(os.path.join("templates", "index.html")), flush=True)
print("=" * 60, flush=True)

app = Flask(__name__)

TASKS = {
    "fetch_stock":   {"script": "fetch_stock.py",   "label": "获取股票数据"},
    "fetch_bond":    {"script": "fetch_bond.py",    "label": "获取债券数据"},
    "analyze_stock": {"script": "analyze_stock.py", "label": "股票均值回归分析"},
    "analyze_bond":  {"script": "analyze_bond.py",  "label": "债券四层决策分析"},
}

STATE = {"running": False, "current": None, "logs": [], "last": None,
         "lock": threading.Lock()}
MAX_LOG_LINES = 2000


def run_task(task_id):
    script = TASKS[task_id]["script"]
    with STATE["lock"]:
        STATE["running"] = True
        STATE["current"] = task_id
        STATE["logs"] = [f">>> 启动：{script}"]
        STATE["last"] = None
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "MPLBACKEND": "Agg"}
    try:
        if not os.path.exists(script):
            raise FileNotFoundError(f"脚本不存在: {script}")
        proc = subprocess.Popen(
            [sys.executable, "-u", script],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1, env=env,
        )
        for line in proc.stdout:
            with STATE["lock"]:
                STATE["logs"].append(line.rstrip())
                if len(STATE["logs"]) > MAX_LOG_LINES:
                    STATE["logs"] = STATE["logs"][-MAX_LOG_LINES:]
        proc.wait()
        with STATE["lock"]:
            STATE["logs"].append(f"<<< 结束：返回码 {proc.returncode}")
            STATE["last"] = {"task": task_id, "returncode": proc.returncode}
    except Exception as e:
        with STATE["lock"]:
            STATE["logs"].append(f"[ERROR] {e}")
            STATE["last"] = {"task": task_id, "error": str(e)}
    finally:
        with STATE["lock"]:
            STATE["running"] = False
            STATE["current"] = None


@app.route("/")
def index():
    return render_template("index.html", tasks=TASKS)


@app.route("/api/start/<task_id>", methods=["POST"])
def start(task_id):
    if task_id not in TASKS:
        return jsonify({"ok": False, "msg": "未知任务"}), 400
    with STATE["lock"]:
        if STATE["running"]:
            return jsonify({"ok": False, "msg": "已有任务在运行"}), 409
    threading.Thread(target=run_task, args=(task_id,), daemon=True).start()
    return jsonify({"ok": True, "msg": f"已启动 {TASKS[task_id]['label']}"})


@app.route("/api/status")
def status():
    with STATE["lock"]:
        return jsonify({"running": STATE["running"], "current": STATE["current"],
                        "logs": STATE["logs"], "last": STATE["last"]})


@app.route("/api/images")
def images():
    result = {}
    for d in ["analysis_output", "analysis_bond"]:
        if os.path.isdir(d):
            result[d] = sorted(f for f in os.listdir(d) if f.lower().endswith(".png"))
        else:
            result[d] = []
    return jsonify(result)


@app.route("/output/<path:subpath>")
def output(subpath):
    for d in ["analysis_output", "analysis_bond"]:
        if os.path.isfile(os.path.join(d, subpath)):
            return send_from_directory(os.path.abspath(d), subpath)
    return "Not Found", 404


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    print("=" * 60, flush=True)
    print(f"启动服务: http://127.0.0.1:{port}", flush=True)
    print("=" * 60, flush=True)
    app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)