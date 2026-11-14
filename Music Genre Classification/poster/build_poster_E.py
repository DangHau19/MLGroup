"""
build_poster_E.py - Sinh poster A1 dọc (3 cột) từ số liệu thật rồi xuất PDF + ảnh xem trước.

Chạy tại thư mục dự án:   python poster/build_poster_E.py
Cần: playwright (Chromium) để xuất PDF; opencv-python để tạo mã QR. Nếu thiếu thì chỉ ghi HTML.
Chỉnh nội dung: sửa phần TEXT bên dưới (hoặc sửa trực tiếp poster_E.html rồi chạy lại bước PDF).
"""
import os, sys, json, shutil, tempfile
from pathlib import Path
import pandas as pd

ROOT = Path(os.environ.get("MG_ROOT", Path(__file__).resolve().parent.parent)).resolve()
OUT = Path(os.environ.get("MG_OUT", ROOT)).resolve()
(OUT / "poster").mkdir(parents=True, exist_ok=True)

# ---- chỉnh ở đây nếu cần ----------------------------------------------------------------------------
KAGGLE_PUBLIC = "0,6885"   # public, từ ảnh Kaggle ngày 15/11/2026
KAGGLE_PRIVATE = "0,6639"
TEAM_A = "A: (Trần Văn Võ Doanh,3124411052 )"
# -----------------------------------------------------------------------------------------------------

E = OUT / "evidence" / "ensemble_E"
if not E.exists():
    E = ROOT / "evidence" / "ensemble_E"
summ = json.loads((E / "summary_E.json").read_text(encoding="utf-8"))
cmp_df = pd.read_csv(E / "strategy_comparison_E.csv").set_index("strategy")
base_df = pd.read_csv(E / "base_models_E.csv")
abl = pd.read_csv(ROOT / "evidence" / "cv_logs" / "ablation_foldsafe_C.csv").set_index("Version")
vn = lambda v, d=4: f"{v:.{d}f}".replace(".", ",")
FINAL = summ["final_strategy"]

rows_res = [("Mốc của A (HistGB v1)", 0.5585), ("CatBoost đơn (D)", float(base_df.loc[0, "cv_accuracy"])),
            ("Trung bình đều 7 mô hình", float(cmp_df.loc["mean_all7", "nested_acc"])),
            ("Trung bình đều 3 boosting", float(cmp_df.loc["mean_boost3", "nested_acc"])),
            ("Stacking HistGB (chốt)", float(cmp_df.loc[FINAL, "nested_acc"]))]
res_html = "".join(f"<tr class='{'hl' if i == len(rows_res) - 1 else ''}'><td>{n}</td><td class='n'>{vn(v)}</td></tr>"
                   for i, (n, v) in enumerate(rows_res))
abl_html = "".join(f"<tr><td>{k}</td><td class='n'>{vn(abl.loc[k, 'Accuracy'])}</td></tr>"
                   for k in ["v0", "v1", "v2", "v2+TE"])
kag = (f"<b>{KAGGLE_PUBLIC}</b>" if KAGGLE_PUBLIC else "<span class='todo'>chờ A nộp bài và điền</span>")
ci = summ["bootstrap_ci95"]


# ---- hình riêng cho poster (chữ lớn hơn hình trong báo cáo) --------------------------------------------
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
PF = OUT / "poster"
y_all = np.load(ROOT / "evidence" / "models_D" / "y_encoded_D.npy")
cnt = np.bincount(y_all, minlength=11)
fig, ax = plt.subplots(figsize=(6.9, 3.9))
ax.bar(range(11), cnt, color="#1f5fa8")
for i, c in enumerate(cnt):
    ax.text(i, c + 60, f"{c:,}".replace(",", "."), ha="center", fontsize=13, rotation=90, va="bottom")
ax.set_xticks(range(11)); ax.tick_params(labelsize=17); ax.set_ylim(0, 5300)
ax.set_xlabel("Thể loại (mã lớp)", fontsize=17); ax.set_ylabel("Số mẫu train", fontsize=17)
fig.tight_layout(); fig.savefig(PF / "fig_p1_class_dist.png", dpi=250); plt.close(fig)

steps = [("A\nmốc", 0.5585), ("C\nv0", abl.loc["v0", "Accuracy"]), ("C\nv1", abl.loc["v1", "Accuracy"]),
         ("C\nv2", abl.loc["v2", "Accuracy"]), ("C\nv2+TE", abl.loc["v2+TE", "Accuracy"]),
         ("D\nđơn", float(base_df.loc[0, "cv_accuracy"])), ("E\nstack", float(cmp_df.loc[FINAL, "nested_acc"]))]
fig, ax = plt.subplots(figsize=(6.9, 3.3))
ax.bar(range(len(steps)), [v for _, v in steps], color=["#9ab6d6"] * 5 + ["#5b8fc7", "#1f5fa8"])
for i, (_, v) in enumerate(steps):
    ax.text(i, v + 0.004, f"{v:.3f}".replace(".", ","), ha="center", fontsize=14)
ax.set_xticks(range(len(steps))); ax.set_xticklabels([n for n, _ in steps], fontsize=14)
ax.set_ylim(0.5, 0.7); ax.tick_params(axis="y", labelsize=15); ax.set_ylabel("CV accuracy", fontsize=17)
fig.tight_layout(); fig.savefig(PF / "fig_p2_progress.png", dpi=250); plt.close(fig)

cmx = pd.read_csv(E / "confusion_matrix_E.csv", index_col=0).values
cmn = cmx / cmx.sum(1, keepdims=True)
fig, ax = plt.subplots(figsize=(6.9, 6.5))
ax.imshow(cmn * 100, cmap="Blues", vmin=0, vmax=100)
for i in range(11):
    for j in range(11):
        v = cmn[i, j] * 100
        ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=14, color="white" if v > 55 else "black")
ax.set_xticks(range(11)); ax.set_yticks(range(11)); ax.tick_params(labelsize=16)
ax.set_xlabel("Thể loại dự đoán", fontsize=18); ax.set_ylabel("Thể loại thật", fontsize=18)
fig.tight_layout(); fig.savefig(PF / "fig_p3_confusion.png", dpi=250); plt.close(fig)

pcd = pd.read_csv(E / "per_class_E.csv")
fig, ax = plt.subplots(figsize=(6.9, 3.8)); x = np.arange(11); w = 0.4
ax.bar(x - w / 2, pcd.f1_single_catboost, w, label="CatBoost đơn", color="#9ab6d6")
ax.bar(x + w / 2, pcd.f1, w, label="Ensemble", color="#1f5fa8")
ax.set_xticks(x); ax.tick_params(labelsize=16); ax.set_ylim(0, 1.18); ax.set_ylabel("F1", fontsize=18)
ax.set_xlabel("Thể loại (mã lớp)", fontsize=17); ax.legend(fontsize=14, ncol=2, loc="upper center", frameon=False)
fig.tight_layout(); fig.savefig(PF / "fig_p4_f1.png", dpi=250); plt.close(fig)

imp = pd.read_csv(ROOT / "evidence" / "feature_importance_E" / "ablation_columns_E.csv")
imp = imp[imp.group.isin(["Artist Name", "duration_in min/ms", "Popularity", "Track Name", "instrumentalness", "acousticness", "tempo"])]
imp = imp.sort_values("delta_accuracy_vs_full")
lab = {"duration_in min/ms": "duration", "Artist Name": "Artist Name", "Track Name": "Track Name"}
fig, ax = plt.subplots(figsize=(6.9, 3.6))
ax.barh([lab.get(g, g) for g in imp.group], imp.delta_accuracy_vs_full * 100, color="#1f5fa8")
for yi, v in enumerate(imp.delta_accuracy_vs_full * 100):
    ax.text(v + 0.2 if v >= 0 else 0.2, yi, f"{v:.1f}".replace(".", ","), va="center", fontsize=15)
ax.tick_params(labelsize=16); ax.set_xlabel("Accuracy giảm khi bỏ cột (điểm %)", fontsize=16); ax.set_xlim(-0.6, 12); ax.set_xticks([0, 3, 6, 9, 12])
fig.tight_layout(); fig.savefig(PF / "fig_p5_importance.png", dpi=250); plt.close(fig)

drop = pd.read_csv(E / "drop_one_model_E.csv")
drop_html = "".join(f"<tr><td>Bỏ {r['bỏ mô hình']}</td><td class='n'>{vn(r.delta_vs_ensemble * 100, 2).replace('-', '−')}</td></tr>"
                    for _, r in drop.sort_values("delta_vs_ensemble").iterrows())

HTML = f"""<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><title>Poster - Music Genre Classification</title>
<style>
@page {{ size: 594mm 841mm; margin: 0 }}
* {{ box-sizing: border-box }}
html, body {{ margin: 0; padding: 0 }}
body {{ width: 594mm; height: 841mm; overflow: hidden; font-family: Inter, 'DejaVu Sans', Arial, sans-serif;
        font-size: 24pt; line-height: 1.28; color: #14213d; background: #fff; display: flex; flex-direction: column }}
header {{ background: #0b2a55; color: #fff; padding: 14mm 18mm 11mm; flex: none }}
h1 {{ font-size: 64pt; line-height: 1.07; margin: 0 0 6mm; font-weight: 800; letter-spacing: -0.5pt }}
.sub {{ font-size: 26pt; color: #cfe0f7; margin: 0 }}
.nw {{ white-space: nowrap }}
.team {{ font-size: 24pt; margin-top: 4mm; color: #fff }}
main {{ flex: 1; display: grid; grid-template-columns: repeat(3, 1fr); gap: 10mm; padding: 10mm 16mm 6mm; min-height: 0 }}
.col {{ display: flex; flex-direction: column; gap: 6mm; min-width: 0 }}
section {{ border-top: 3mm solid #1f5fa8; padding-top: 3mm }}
h2 {{ font-size: 34pt; margin: 0 0 3mm; color: #0b2a55; line-height: 1.1 }}
p {{ margin: 0 0 3mm }}
ul {{ margin: 0; padding-left: 8mm }} li {{ margin-bottom: 2.5mm }}
img {{ width: 100%; display: block }}
.cap {{ font-size: 24pt; color: #43506b; margin: 1mm 0 2mm }}
table {{ border-collapse: collapse; width: 100%; font-size: 24pt }}
td {{ border-bottom: 0.6mm solid #cbd5e6; padding: 1.4mm 2mm }} td.n {{ text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap }}
tr.hl td {{ background: #e3eefc; font-weight: 700 }}
.flow {{ display: flex; flex-direction: column; gap: 2.5mm; margin: 2mm 0 3mm }}
.box {{ background: #eaf1fb; border-left: 3mm solid #1f5fa8; padding: 2mm 4mm }} .box b {{ color: #0b2a55 }}
.arrow {{ text-align: center; color: #1f5fa8; line-height: .6 }}
.todo {{ background: #fff4c2; border: 0.8mm dashed #b58900; padding: 0 2mm }}
footer {{ flex: none; display: flex; align-items: center; gap: 10mm; padding: 5mm 16mm; background: #eef3fa; border-top: 1mm solid #1f5fa8 }}
footer img.qr {{ width: 46mm; height: 46mm }} footer div {{ flex: 1 }} footer .u {{ word-break: break-all; font-size: 24pt }}
</style></head><body>
<header>
  <h1>Phân loại thể loại nhạc từ đặc trưng dạng bảng: mã hóa theo nghệ sĩ và ensemble</h1>
  <p class="sub">Challenge 2 · Môn Máy học · Cuộc thi Kaggle “shai-music-genre-classification”</p>
  <p class="team"><span class="nw">{TEAM_A}</span> · <span class="nw">B: Nguyễn Minh Triết (3124411320)</span> · <span class="nw">C: Phan Lê Minh Huy (3124411111)</span> ·
  <span class="nw">D: Trần Đình Bảo Quốc (3124411252)</span> · <span class="nw">E: Hoàng Đăng Hậu (3124411085)</span></p>
</header>
<main>
 <div class="col">
  <section><h2>1. Bài toán và dữ liệu</h2>
   <ul><li>Dự đoán <b>11 thể loại</b> (nhãn 0–10) của bài hát từ bảng đặc trưng kiểu Spotify; không có tệp âm thanh.</li>
   <li><b>14.396</b> mẫu train, <b>3.600</b> mẫu test; thang chấm của cuộc thi: <b>accuracy</b>.</li>
   <li>Mất cân bằng: lớp 10 có 3.959 mẫu, lớp 4 chỉ 310 mẫu (lệch 12,8 lần).</li></ul>
   <img src="fig_p1_class_dist.png" alt="Hình 1">
   <div class="cap">Hình 1. Phân bố 11 lớp trong tập train.</div>
  </section>
  <section><h2>2. Phát hiện từ EDA</h2>
   <ul><li><b>Thời lượng lẫn hai đơn vị:</b> 14,4% dòng (2.075) tính bằng phút, còn lại bằng mili giây.</li>
   <li><b>Thiếu dữ liệu có quy luật:</b> instrumentalness thiếu 24,6%, key 11,2%; tỉ lệ thiếu khác nhau theo thể loại.</li>
   <li><b>Nghệ sĩ có giá trị:</b> 63,8% dòng test có nghệ sĩ đã xuất hiện ở train; nhiều nghệ sĩ có hơn một thể loại.</li></ul>
  </section>
  <section><h2>3. Tiền xử lý và đặc trưng</h2>
   <ul><li>Quy đổi thời lượng về một thang, giữ cờ “thuộc thang phút” và cờ thiếu dữ liệu.</li>
   <li>Biến đổi log, tỉ số và tương tác giữa các đặc trưng (v2).</li>
   <li><b>Artist Target Encoding</b> có làm mịn, tính trong từng fold với OOF lồng bên trong để không rò rỉ nhãn.</li></ul>
   <table><tr><td><b>Phiên bản đặc trưng</b></td><td class="n"><b>CV acc.</b></td></tr>{abl_html}</table>
   <div class="cap">Bảng 1. CV accuracy của HistGB theo phiên bản đặc trưng (C): thông tin nghệ sĩ là bước nhảy lớn nhất.</div>
  </section>
 </div>

 <div class="col">
  <section><h2>4. Phương pháp</h2>
   <div class="flow">
    <div class="box"><b>B</b> · EDA, kiểm chứng dữ liệu</div><div class="arrow">▼</div>
    <div class="box"><b>C</b> · Đặc trưng v0–v2 + Artist Target Encoding</div><div class="arrow">▼</div>
    <div class="box"><b>D</b> · 7 mô hình, tuning Optuna, 5-fold phân tầng (seed 42)</div><div class="arrow">▼</div>
    <div class="box"><b>E</b> · Ensemble và stacking trên xác suất OOF</div>
   </div>
   <ul><li><b>Mô hình nền:</b> CatBoost, LightGBM, XGBoost, HistGB, MLP, SVM, ExtraTrees.</li>
   <li><b>Ensemble:</b> trung bình, trọng số tối ưu, chọn tham lam, stacking (logistic và HistGB).</li>
   <li><b>Đánh giá nested:</b> học trọng số/meta-learner trên 4 fold OOF, chấm trên fold còn lại.</li></ul>
  </section>
  <section><h2>5. Kết quả</h2>
   <table><tr><td><b>Mô hình / chiến lược</b></td><td class="n"><b>CV acc.</b></td></tr>{res_html}</table>
   <div class="cap">Bảng 2. CV accuracy (5 fold). Điểm ensemble là nested.</div>
   <img src="fig_p2_progress.png" alt="Hình 2">
   <div class="cap">Hình 2. Accuracy CV qua các giai đoạn (trục bắt đầu từ 0,5).</div>
   <ul><li>Stacking hơn CatBoost đơn <b>+{vn(summ['diff_vs_single'] * 100, 2)} điểm %</b> (CI 95%: +{vn(ci[0] * 100, 2)} đến +{vn(ci[1] * 100, 2)}), hơn ở cả 5 fold.</li>
   <li><b>Kaggle:</b> public {kag}, private <b>{KAGGLE_PRIVATE}</b> (nộp muộn), sát với nested CV 0,6623.</li></ul>
  </section>
  <section><h2>8. Kết luận và hạn chế</h2>
   <ul><li>Thông tin nghệ sĩ cải thiện nhiều hơn mọi lựa chọn mô hình; ensemble thêm khoảng 0,3–0,8 điểm %.</li>
   <li>Trung bình đều 7 mô hình thấp hơn mô hình đơn tốt nhất.</li>
   <li>Private 0,664 khớp CV 0,662; chưa kiểm tra nghệ sĩ mới.</li></ul>
  </section>
 </div>

 <div class="col">
  <section><h2>6. Phân tích lỗi</h2>
   <img src="fig_p3_confusion.png" alt="Hình 3">
   <div class="cap">Hình 3. Ma trận nhầm lẫn (% theo hàng).</div>
   <ul><li><b>Lớp ít mẫu không bị bỏ qua:</b> lớp 3 và 4 có F1 = 0,85 và 0,83; lớp 7 đạt 0,95.</li>
   <li><b>Khó nhất:</b> lớp 1 (F1 0,31), lớp 6 (0,54); thường bị nhầm sang lớp 10, lớp lớn nhất.</li>
   <li><b>ExtraTrees</b> yếu nhất khi đứng riêng ({vn(float(base_df.set_index('model').loc['ExtraTrees', 'cv_accuracy']))}) nhưng bỏ nó đi thì ensemble giảm nhiều nhất (−0,85 điểm %): giá trị nằm ở sự đa dạng.</li>
   </ul>
   <img src="fig_p4_f1.png" alt="Hình 4">
   <div class="cap">Hình 4. F1 theo thể loại: ensemble so với CatBoost đơn.</div>
  </section>
  <section><h2>7. Cột nào quan trọng?</h2>
   <img src="fig_p5_importance.png" alt="Hình 5">
   <div class="cap">Hình 5. Bỏ từng cột rồi huấn luyện lại (HistGB, 5 fold).</div>
   <ul><li><b>Artist Name</b> quan trọng nhất (−10,0 điểm %), rồi <b>duration</b> (−4,0).</li>
   <li>Bỏ duration, F1 lớp 0 tụt từ 0,81 xuống 0,43: cờ đơn vị phút/ms tách nhóm lớp {{0, 3, 4, 7}}.</li></ul>
  </section>
 </div>
</main>
<footer></footer>
</body></html>"""
(OUT / "poster" / "poster_E.html").write_text(HTML, encoding="utf-8")
print("Đã ghi poster/poster_E.html")


# ---- PDF + ảnh xem trước --------------------------------------------------------------------------------
try:
    from playwright.sync_api import sync_playwright
except Exception as e:                                   # noqa: BLE001
    print("Không có playwright, bỏ qua bước PDF:", e); sys.exit(0)
tmp = Path(tempfile.mkdtemp())
(tmp / "poster").mkdir(); (tmp / "paper" / "figures").mkdir(parents=True)
for f in (OUT / "poster").glob("*"):
    if f.is_file(): shutil.copy(f, tmp / "poster" / f.name)
for src in (ROOT / "paper" / "figures", OUT / "paper" / "figures"):
    for f in src.glob("*.png"): shutil.copy(f, tmp / "paper" / "figures" / f.name)
with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page()
    pg.goto((tmp / "poster" / "poster_E.html").as_uri()); pg.wait_for_load_state("networkidle")
    audit = pg.evaluate("""() => {
      const out = {minPx: 1e9, small: [], title: 0};
      const w = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
      while (w.nextNode()) { const t = w.currentNode; if (!t.textContent.trim()) continue;
        const px = parseFloat(getComputedStyle(t.parentElement).fontSize); out.minPx = Math.min(out.minPx, px);
        if (px < 32 - 0.01) out.small.push(t.textContent.trim().slice(0, 40)); }
      out.title = parseFloat(getComputedStyle(document.querySelector('h1')).fontSize);
      const m = document.querySelector('main');
      out.overflow = [...document.querySelectorAll('.col')].map(c => c.scrollHeight - c.clientHeight);
      out.bodyScroll = document.body.scrollHeight - document.body.clientHeight;
      return out; }""")
    print("Kiểm tra cỡ chữ: nhỏ nhất =", round(audit["minPx"] * 0.75, 1), "pt; tiêu đề =", round(audit["title"] * 0.75, 1),
          "pt; đoạn nhỏ hơn 24pt:", audit["small"], "; tràn cột (px):", audit["overflow"], "; tràn trang (px):", audit["bodyScroll"])
    pg.pdf(path=str(OUT / "poster" / "poster_E.pdf"), width="594mm", height="841mm", print_background=True,
           prefer_css_page_size=True)
    pg.set_viewport_size({"width": 2246, "height": 3180})
    prev = OUT / "poster" / "poster_E_preview.png"; pg.screenshot(path=str(prev)); b.close()
try:                                                      # thu nhỏ ảnh xem trước cho nhẹ
    from PIL import Image
    im = Image.open(prev); im.thumbnail((1200, 1700)); im.save(prev)
except Exception:                                         # noqa: BLE001
    pass
print("Đã ghi poster/poster_E.pdf và poster_E_preview.png")
