"""Face attendance system.  Run: uvicorn app:app --host 0.0.0.0 --port 8000  ->  http://localhost:8000"""
import io, os, sqlite3, threading, time
from collections import defaultdict
from datetime import datetime

import cv2, numpy as np, pandas as pd
from fastapi import Body, FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from insightface.app import FaceAnalysis

THRESH, MARGIN, MIN_DET, MIN_FACE, BLUR_MIN, STABLE = 0.45, 0.08, 0.60, 80, 60, 3
DEMO = os.getenv("DEMO") == "1"      # public portfolio mode: sample data + student cap
DEMO_MAX = 20
os.makedirs("exports", exist_ok=True)
MODEL = os.getenv("MODEL", "buffalo_l")   # buffalo_l = most accurate; buffalo_sc = tiny, fits 512 MB free hosts
DET = int(os.getenv("DET", "640"))         # detector size; 320 is faster on weak CPUs
fa = FaceAnalysis(name=MODEL, allowed_modules=["detection", "recognition"], providers=["CPUExecutionProvider"])
fa.prepare(ctx_id=-1, det_size=(DET, DET))

db = sqlite3.connect("attendance.db", check_same_thread=False)
L = threading.RLock()
db.executescript("""
CREATE TABLE IF NOT EXISTS students(roll TEXT PRIMARY KEY, name TEXT, grp TEXT DEFAULT '');
CREATE TABLE IF NOT EXISTS emb(roll TEXT, v BLOB);
CREATE TABLE IF NOT EXISTS att(day TEXT, t TEXT, roll TEXT, sid INTEGER, status TEXT, conf REAL, PRIMARY KEY(day, roll, sid));
CREATE TABLE IF NOT EXISTS sched(id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, days TEXT, start TEXT, end TEXT, late TEXT, on_ INTEGER DEFAULT 1);
CREATE TABLE IF NOT EXISTS settings(k TEXT PRIMARY KEY, v TEXT);
""")
try: db.execute("ALTER TABLE students ADD COLUMN grp TEXT DEFAULT ''")
except sqlite3.OperationalError: pass


def q(sql, a=()):
    with L: return db.execute(sql, a).fetchall()

def run(sql, a=()):
    with L:
        c = db.execute(sql, a); db.commit(); return c


G_V, G_R = np.zeros((0, 512), np.float32), []
def load_gallery():
    global G_V, G_R
    rows = q("SELECT roll, v FROM emb")
    G_R = [r for r, _ in rows]
    G_V = np.array([np.frombuffer(v, np.float32) for _, v in rows]).reshape(-1, 512)
load_gallery()
streak = defaultdict(int)
app = FastAPI()
decode = lambda b: cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
faces_in = lambda img: [f for f in fa.get(img) if f.det_score >= MIN_DET and f.bbox[2] - f.bbox[0] >= MIN_FACE]


def identify(emb):
    if not len(G_R): return None, 0.0
    per = {}
    for r, s in zip(G_R, G_V @ emb): per[r] = max(per.get(r, -1.0), float(s))
    rk = sorted(per.items(), key=lambda x: -x[1])
    second = rk[1][1] if len(rk) > 1 else -1.0
    return (rk[0][0], rk[0][1]) if rk[0][1] >= THRESH and rk[0][1] - second >= MARGIN else (None, rk[0][1])


# ---------- schedule: decides when attendance is open ----------
def mode():
    r = q("SELECT v FROM settings WHERE k='mode'"); return r[0][0] if r else "auto"

def active():
    m, now = mode(), datetime.now()
    if m == "off": return None
    if m == "on": return {"id": 0, "name": "Manual session", "late": "23:59", "end": "23:59"}
    hm, wd = now.strftime("%H:%M"), str(now.weekday())
    for i, n, d, s, e, l, en in q("SELECT id,name,days,start,end,late,on_ FROM sched"):
        if en and wd in d.split(",") and s <= hm < e: return {"id": i, "name": n, "late": l, "end": e}
    return None

@app.get("/api/status")
def status():
    now = datetime.now(); hm, wd = now.strftime("%H:%M"), str(now.weekday())
    nxt = sorted((s, n) for i, n, d, s, e, l, en in q("SELECT id,name,days,start,end,late,on_ FROM sched") if en and wd in d.split(",") and s > hm)
    return {"demo": DEMO, "mode": mode(), "active": active(), "next": f"{nxt[0][0]} {nxt[0][1]}" if nxt else ""}

@app.post("/api/mode")
def set_mode(d: dict = Body(...)):
    if d.get("mode") in ("auto", "on", "off"): run("INSERT OR REPLACE INTO settings VALUES('mode',?)", (d["mode"],))
    return status()

@app.get("/api/schedules")
def schedules():
    return [dict(id=i, name=n, days=d, start=s, end=e, late=l, on=bool(en)) for i, n, d, s, e, l, en in q("SELECT id,name,days,start,end,late,on_ FROM sched ORDER BY start")]

@app.post("/api/schedules")
def save_schedule(d: dict = Body(...)):
    v = (d["name"], d["days"], d["start"], d["end"], d.get("late") or d["end"], int(d.get("on", True)))
    if d.get("id"): run("UPDATE sched SET name=?,days=?,start=?,end=?,late=?,on_=? WHERE id=?", v + (d["id"],))
    else: run("INSERT INTO sched(name,days,start,end,late,on_) VALUES(?,?,?,?,?,?)", v)
    return {"ok": True}

@app.delete("/api/schedules/{sid}")
def del_schedule(sid: int):
    run("DELETE FROM sched WHERE id=?", (sid,)); return {"ok": True}


# ---------- students ----------
@app.post("/api/enroll")
async def enroll(roll: str = Form(...), name: str = Form(...), grp: str = Form(""), replace: int = Form(0), files: list[UploadFile] = File(...)):
    roll, name = roll.strip(), " ".join(name.split())
    if not roll or not name: return {"error": "Roll number and name are required."}
    if q("SELECT 1 FROM students WHERE roll=?", (roll,)) and not replace: return {"error": f"Roll {roll} is already registered."}
    if DEMO and not q("SELECT 1 FROM students WHERE roll=?", (roll,)) and q("SELECT COUNT(*) FROM students")[0][0] >= DEMO_MAX:
        return {"error": f"Demo limit reached ({DEMO_MAX} students). Delete one to add another."}
    embs = []
    for f in files:
        img = decode(await f.read())
        if img is None or cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var() < BLUR_MIN: continue
        fs = faces_in(img)
        if len(fs) == 1: embs.append(fs[0].normed_embedding.astype(np.float32))
    if len(embs) < 3: return {"error": f"Only {len(embs)} usable photos. Need at least 3, each with one sharp face in good light."}
    for e in embs:
        other, s = identify(e)
        if other and other != roll: return {"error": f"This face already matches roll {other}."}
    with L:
        db.execute("DELETE FROM emb WHERE roll=?", (roll,))
        db.executemany("INSERT INTO emb VALUES(?,?)", [(roll, e.tobytes()) for e in embs])
        db.execute("INSERT OR REPLACE INTO students VALUES(?,?,?)", (roll, name, grp.strip())); db.commit()
    load_gallery()
    return {"accepted": len(embs), "received": len(files)}

@app.get("/api/students")
def students(qs: str = "", grp: str = ""):
    return [dict(roll=r, name=n, grp=g or "", photos=c) for r, n, g, c in q(
        "SELECT s.roll,s.name,s.grp,(SELECT COUNT(*) FROM emb e WHERE e.roll=s.roll) FROM students s "
        "WHERE (s.name LIKE ? OR s.roll LIKE ?) AND (?='' OR s.grp=?) ORDER BY s.name", (f"%{qs}%", f"%{qs}%", grp, grp))]

@app.put("/api/students/{roll}")
def edit_student(roll: str, d: dict = Body(...)):
    run("UPDATE students SET name=?, grp=? WHERE roll=?", (d["name"], d.get("grp", ""), roll)); return {"ok": True}

@app.delete("/api/students/{roll}")
def del_student(roll: str):
    for t in ("emb", "att", "students"): run(f"DELETE FROM {t} WHERE roll=?", (roll,))
    load_gallery(); return {"ok": True}


# ---------- live recognition ----------
@app.post("/api/recognize")
async def recognize(file: UploadFile = File(...)):
    act = active()
    if not act: return {"active": False, "faces": []}
    img, now, out, seen = decode(await file.read()), datetime.now(), [], set()
    for f in faces_in(img):
        roll, s = identify(f.normed_embedding)
        item = {"box": [int(x) for x in f.bbox], "name": "Unknown", "roll": None, "sim": round(s, 3), "marked": ""}
        if roll:
            seen.add(roll); streak[roll] += 1
            item["name"], item["roll"] = q("SELECT name FROM students WHERE roll=?", (roll,))[0][0], roll
            if streak[roll] >= STABLE:
                st = "Late" if now.strftime("%H:%M") > act["late"] else "Present"
                c = run("INSERT OR IGNORE INTO att VALUES(?,?,?,?,?,?)", (now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"), roll, act["id"], st, s))
                item["marked"] = st if c.rowcount == 1 else ""
        out.append(item)
    for r in list(streak):
        if r not in seen: streak[r] = 0
    return {"active": True, "faces": out}


# ---------- dashboard ----------
@app.get("/api/summary")
def summary():
    day = datetime.now().strftime("%Y-%m-%d")
    total = q("SELECT COUNT(*) FROM students")[0][0]
    present = q("SELECT COUNT(DISTINCT roll) FROM att WHERE day=?", (day,))[0][0]
    late = q("SELECT COUNT(DISTINCT roll) FROM att WHERE day=? AND status='Late'", (day,))[0][0]
    return {"total": total, "present": present, "late": late, "absent": max(total - present, 0)}

@app.get("/api/log")
def log():
    return [dict(t=t, roll=r, name=n, status=s, session=c) for t, r, n, s, c in q(
        "SELECT a.t,a.roll,s.name,a.status,COALESCE(c.name,'Manual') FROM att a JOIN students s USING(roll) "
        "LEFT JOIN sched c ON c.id=a.sid WHERE a.day=? ORDER BY a.t DESC", (datetime.now().strftime("%Y-%m-%d"),))]

@app.get("/api/chart")
def chart(days: int = 14):
    total = max(q("SELECT COUNT(*) FROM students")[0][0], 1)
    ds = [d[0] for d in q("SELECT DISTINCT day FROM att ORDER BY day DESC LIMIT ?", (days,))][::-1]
    return [{"d": d[5:], "p": round(100 * q("SELECT COUNT(DISTINCT roll) FROM att WHERE day=?", (d,))[0][0] / total)} for d in ds]


# ---------- reports + Excel ----------
def report(start="", end="", grp="", status="", qs=""):
    today = datetime.now().strftime("%Y-%m-%d")
    start, end = start or today[:8] + "01", end or today
    st = pd.DataFrame(q("SELECT roll,name,grp FROM students"), columns=["Roll", "Name", "Group"])
    if grp: st = st[st.Group == grp]
    if qs: st = st[st.Name.str.contains(qs, case=False, regex=False) | st.Roll.str.contains(qs, case=False, regex=False)]
    at = pd.DataFrame(q("SELECT a.day,a.t,a.roll,a.status,COALESCE(c.name,'Manual') FROM att a LEFT JOIN sched c ON c.id=a.sid "
                        "WHERE a.day BETWEEN ? AND ?", (start, end)), columns=["Date", "Time", "Roll", "Status", "Session"])
    df = at[["Date", "Session"]].drop_duplicates().merge(st, how="cross").merge(at, on=["Date", "Session", "Roll"], how="left")
    df["Status"], df["Time"] = df["Status"].fillna("Absent"), df["Time"].fillna("")
    return df[df.Status == status] if status else df    # absences are counted only on days a session actually ran

def shape(df, kind):
    if kind == "daily": return df[["Date", "Session", "Roll", "Name", "Group", "Time", "Status"]].sort_values(["Date", "Session", "Name"]).reset_index(drop=True)
    keys = ["Roll", "Name", "Group"] if kind == "student" else ["Date", "Session"]
    if df.empty: return pd.DataFrame(columns=keys + ["Present", "Late", "Absent", "Total", "Rate %"])
    g = pd.crosstab([df[k] for k in keys], df.Status).reset_index().reindex(columns=keys + ["Present", "Late", "Absent"], fill_value=0)
    g["Total"] = g.Present + g.Late + g.Absent
    g["Rate %"] = ((g.Present + g.Late) * 100 / g.Total).round(0).astype(int)
    return g

@app.get("/api/report")
def api_report(kind: str = "daily", start: str = "", end: str = "", grp: str = "", status: str = "", qs: str = ""):
    df = report(start, end, grp, status, qs)
    tot = {k: int(v) for k, v in df.Status.value_counts().items()}
    return {"rows": shape(df, kind).fillna("").to_dict("records"), "totals": tot}

def workbook(**f):
    df, buf = report(**f), io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        for name, k in (("Records", "daily"), ("By student", "student"), ("By day", "day")):
            shape(df, k).to_excel(w, sheet_name=name, index=False)
    return buf.getvalue()

@app.get("/api/export")
def export(start: str = "", end: str = "", grp: str = "", status: str = "", qs: str = "", kind: str = ""):
    data = workbook(start=start, end=end, grp=grp, status=status, qs=qs)
    return StreamingResponse(io.BytesIO(data), media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                             headers={"Content-Disposition": f'attachment; filename="attendance_{start or "month"}_{end or "today"}.xlsx"'})

def auto_export():
    # after each scheduled session ends, save that day's workbook into ./exports
    done = set()
    while True:
        time.sleep(30)
        try:
            now = datetime.now(); hm, d, wd = now.strftime("%H:%M"), now.strftime("%Y-%m-%d"), str(now.weekday())
            for i, n, days, s, e, l, en in q("SELECT id,name,days,start,end,late,on_ FROM sched"):
                if en and wd in days.split(",") and hm >= e and (d, i) not in done:
                    done.add((d, i))
                    if q("SELECT 1 FROM att WHERE day=? AND sid=? LIMIT 1", (d, i)):
                        open(f"exports/attendance_{d}_{n.replace(' ', '_')}.xlsx", "wb").write(workbook(start=d, end=d))
        except Exception as ex: print("auto-export error:", ex)
threading.Thread(target=auto_export, daemon=True).start()

def seed_demo():
    # sample data so the dashboard and reports look alive on a fresh public demo
    if q("SELECT COUNT(*) FROM students")[0][0]: return
    import random; from datetime import timedelta
    random.seed(7)
    run("INSERT INTO sched(name,days,start,end,late) VALUES('Morning class','0,1,2,3,4','08:00','09:30','08:15')")
    sid = q("SELECT id FROM sched")[0][0]
    names = ["Abel Tesfaye", "Beza Alemu", "Chala Bekele", "Dawit Kebede", "Eden Girma", "Fitsum Haile", "Genet Tadesse", "Hana Mekonnen", "Kaleb Worku", "Liya Assefa", "Meron Demissie", "Nahom Tsegaye"]
    for i, n in enumerate(names): run("INSERT INTO students VALUES(?,?,?)", (str(i + 1), n, ["CS-3A", "CS-3B"][i % 2]))
    for back in range(1, 29):
        day = datetime.now() - timedelta(days=back)
        if day.weekday() > 4: continue
        for i in range(len(names)):
            r = random.random()
            if r < 0.12: continue
            late = r > 0.85
            t = f"08:{random.randint(16, 40) if late else random.randint(0, 14):02d}:{random.randint(0, 59):02d}"
            run("INSERT OR IGNORE INTO att VALUES(?,?,?,?,?,?)", (day.strftime("%Y-%m-%d"), t, str(i + 1), sid, "Late" if late else "Present", 0.7))
if DEMO: seed_demo()

@app.get("/")
def index(): return FileResponse("dashboard.html")
