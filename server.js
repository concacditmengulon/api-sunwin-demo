/**
 * API Taixiu SIÊU VIP – by Tele@idol_vannhat
 * Node >= 18 (có fetch sẵn)
 * Endpoint: GET /api/custom
 * Trả về:
 * {
 *   id: "Tele@idol_vannhat",
 *   Phien_truoc, Phien_sau,
 *   Xuc_xac: [x1,x2,x3],
 *   Tong, Ket_qua, Du_doan, Do_tin_cay, Giai_thich, Mau_cau
 * }
 */

import express from "express";

const app = express();
const PORT = process.env.PORT || 3000;

// ============= Cấu hình nguồn dữ liệu =============
const SOURCE_API = "https://sunai.onrender.com/api/taixiu/history";

// ============= Bộ nhớ trong server =============
let history = [];            // [{session, dice:[a,b,c], total, result}]
let patternMemory = {};      // n-gram 'T'/'X' -> đếm tần suất tiếp theo
let modelPredictions = {};   // {session: {module: "Tài"/"Xỉu", ... , final}}
const MAX_HISTORY = 400;

// ============= Tiện ích chung =============
const r01 = (x) => Math.round(x * 100) / 100;
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
const last = (arr, k = 1) => arr.slice(-k);
const toTX = (res) => (res === "Tài" ? "T" : "X");
const fromTX = (c) => (c === "T" ? "Tài" : "Xỉu");

// ============= Nạp & chuẩn hoá dữ liệu =============
async function loadSource() {
  const resp = await fetch(SOURCE_API, { cache: "no-store" });
  const data = await resp.json();
  if (!Array.isArray(data) || data.length === 0) {
    throw new Error("API nguồn không trả mảng dữ liệu hợp lệ");
  }
  const norm = data
    .filter(x => x && typeof x.session !== "undefined")
    .map(x => ({
      session: Number(x.session),
      dice: Array.isArray(x.dice) ? x.dice.map(Number) : [0, 0, 0],
      total: Number(x.total ?? 0),
      result: x.result === "Tài" || x.result === "Xỉu"
        ? x.result
        : (Number(x.total ?? 0) >= 11 ? "Tài" : "Xỉu")
    }))
    .sort((a, b) => a.session - b.session);

  const seen = new Set(history.map(h => h.session));
  for (const row of norm) if (!seen.has(row.session)) history.push(row);
  if (history.length > MAX_HISTORY) history = history.slice(-MAX_HISTORY);

  rebuildPatternMemory();
}

function rebuildPatternMemory() {
  patternMemory = {};
  const seq = history.map(h => toTX(h.result)); // 'T'/'X'
  const N = seq.length;
  const ns = [3, 4, 5];
  for (const n of ns) {
    for (let i = 0; i <= N - (n + 1); i++) {
      const key = seq.slice(i, i + n).join("");
      const nxt = seq[i + n];
      if (!patternMemory[n]) patternMemory[n] = {};
      if (!patternMemory[n][key]) patternMemory[n][key] = { T: 0, X: 0 };
      patternMemory[n][key][nxt]++;
    }
  }
}

// ============= Phân tích nền tảng =============
function detectStreak(hist) {
  if (hist.length === 0) return { streak: 0, current: null };
  const current = hist.at(-1).result;
  let streak = 1;
  for (let i = hist.length - 2; i >= 0; i--) {
    if (hist[i].result === current) streak++;
    else break;
  }
  return { streak, current };
}

function breakProbability(hist) {
  const { streak, current } = detectStreak(hist);
  const last20 = hist.slice(-20).map(x => x.result);
  const switches = last20.slice(1).reduce((c, r, i) => c + (r !== last20[i] ? 1 : 0), 0);
  const taiCount = last20.filter(r => r === "Tài").length;
  const pT = last20.length ? taiCount / last20.length : 0.5;
  const pX = 1 - pT;
  const entropy = -(pT ? pT * Math.log2(pT) : 0) - (pX ? pX * Math.log2(pX) : 0);

  let base =
    streak >= 9 ? 0.78 :
    streak >= 7 ? 0.6 :
    streak >= 5 ? 0.45 :
    streak >= 4 ? 0.35 : 0.25;

  base += switches / 50;           // càng đảo chiều nhiều càng dễ bẻ
  base += entropy < 0.85 ? 0.08 : 0; // thiên lệch mạnh → dễ bẻ

  return clamp(base, 0.05, 0.96);
}

function markovTransition(hist) {
  if (hist.length < 6) return { pT: 0.5, pX: 0.5 };
  const M = { Tài: { Tài: 0, Xỉu: 0 }, Xỉu: { Tài: 0, Xỉu: 0 } };
  for (let i = 1; i < hist.length; i++) M[hist[i - 1].result][hist[i].result]++;
  const cur = hist.at(-1).result;
  const a = M[cur].Tài + 1; // Laplace
  const b = M[cur].Xỉu + 1;
  const pT = a / (a + b);
  return { pT, pX: 1 - pT };
}

function totalsStats(hist, k = 12) {
  const v = hist.slice(-k).map(x => x.total || 0);
  const avg = v.reduce((s, x) => s + x, 0) / (v.length || 1);
  const varc = v.reduce((s, x) => s + (x - avg) ** 2, 0) / (v.length || 1);
  return { avg, varc };
}

// ============= Phân tích Mau_cau-15 =============
/**
 * Dùng 15 ký tự T/X gần nhất để ra P(next=T/X)
 * - n-gram context 5,4,3 với Laplace smoothing
 * - run-length & alternating penalty
 * - entropy control
 */
function mauCau15Predict() {
  if (history.length < 16) return { pred: null, conf: 0.0, note: "Thiếu lịch sử" };

  const seq = history.map(h => toTX(h.result));
  const MC = seq.slice(-15);           // 15 ký tự cuối
  const window5 = seq.slice(-5).join(""); // phục vụ lookup patternMemory
  const contexts = [
    { n: 5, key: seq.slice(-5).join("") },
    { n: 4, key: seq.slice(-4).join("") },
    { n: 3, key: seq.slice(-3).join("") },
  ];

  // n-gram Bayes
  let pT = 0.5, used = null;
  for (const ctx of contexts) {
    const mem = patternMemory[ctx.n]?.[ctx.key];
    if (mem) {
      const aT = mem.T + 1; // Laplace
      const aX = mem.X + 1;
      const post = aT / (aT + aX);
      pT = 0.6 * post + 0.4 * pT; // gộp mềm
      used = `${ctx.n}-gram ${ctx.key}→T:${mem.T}/X:${mem.X}`;
    }
  }

  // run-length trên chuỗi 15
  let r = 1, lastC = MC[0], maxR = 1;
  for (let i = 1; i < MC.length; i++) {
    if (MC[i] === lastC) { r++; maxR = Math.max(maxR, r); }
    else { r = 1; lastC = MC[i]; }
  }
  if (maxR >= 6) {
    // chuỗi vừa qua rất dài → tăng xác suất đảo
    const lastChar = MC.at(-1);
    if (lastChar === "T") pT = clamp(pT - 0.1, 0.04, 0.96);
    else pT = clamp(pT + 0.1, 0.04, 0.96);
  }

  // alternating mạnh (TXTX...) → ưu tiên giữ nhịp
  const sw = MC.slice(1).reduce((c, ch, i) => c + (ch !== MC[i] ? 1 : 0), 0);
  if (sw >= 11) {
    const lastChar = MC.at(-1);
    pT = lastChar === "T" ? clamp(pT - 0.08, 0.04, 0.96) : clamp(pT + 0.08, 0.04, 0.96);
  }

  // entropy của 15
  const cntT = MC.filter(c => c === "T").length;
  const pT15 = cntT / 15;
  const pX15 = 1 - pT15;
  const H = -(pT15 ? pT15 * Math.log2(pT15) : 0) - (pX15 ? pX15 * Math.log2(pX15) : 0);
  let confBoost = H < 0.85 ? 0.08 : 0; // thiên lệch → tự tin hơn 1 chút

  const pred = pT >= 0.5 ? "Tài" : "Xỉu";
  const conf = clamp(0.55 + Math.abs(pT - 0.5) * 0.9 + confBoost, 0.52, 0.96);

  return { pred, conf, note: used ? `Mau_cau15: ${used}, pT=${r01(pT)}` : `Mau_cau15: pT=${r01(pT)}` };
}

// ============= AI phân tích cầu PRO (full nâng cao) =============
function aiProAnalyzer(hist) {
  if (hist.length < 7) return { pred: Math.random() < 0.5 ? "Tài" : "Xỉu", reason: "Thiếu dữ liệu" };

  const { streak, current } = detectStreak(hist);
  const bprob = breakProbability(hist);
  const { avg, varc } = totalsStats(hist, 14);
  const last8 = hist.slice(-8).map(x => x.result);

  // detector “bậc thang”: ví dụ T,T,X,X,T,T → nghi nghiêng về đảo
  let stair = 0;
  for (let i = 2; i < last8.length; i++) {
    if (last8[i] === last8[i - 1] && last8[i - 2] !== last8[i - 1]) stair++;
  }

  // alternating mạnh
  const sw8 = last8.slice(1).reduce((c, r, i) => c + (r !== last8[i] ? 1 : 0), 0);

  let scoreBreak = 0;
  scoreBreak += (streak >= 7) ? 0.35 : (streak >= 5 ? 0.2 : 0.08);
  scoreBreak += bprob * 0.6;
  scoreBreak += (varc > 9 ? 0.1 : 0) + (stair >= 2 ? 0.06 : 0) + (sw8 >= 6 ? 0.05 : 0);

  // drift quanh 10.5
  if (avg > 10.8) scoreBreak += current === "Tài" ? -0.04 : 0.06;
  if (avg < 10.2) scoreBreak += current === "Xỉu" ? -0.04 : 0.06;

  scoreBreak = clamp(scoreBreak, 0, 0.95);
  const pred = scoreBreak > 0.58 ? (current === "Tài" ? "Xỉu" : "Tài") : current;

  return {
    pred,
    reason: `Streak=${streak}, BreakProb=${r01(bprob)}, avg=${r01(avg)}, var=${r01(varc)}, stair=${stair}, sw8=${sw8}`
  };
}

// ============= Các module phụ khác (giữ & tinh chỉnh) =============
function trendWeighted(hist) {
  if (hist.length < 6) return Math.random() < 0.5 ? "Tài" : "Xỉu";
  const last20 = hist.slice(-20).map(x => x.result);
  const w = last20.map((_, i) => Math.pow(1.25, i));
  const tW = w.reduce((s, ww, i) => s + (last20[i] === "Tài" ? ww : 0), 0);
  const xW = w.reduce((s, ww, i) => s + (last20[i] === "Xỉu" ? ww : 0), 0);
  if (Math.abs(tW - xW) / (tW + xW) >= 0.28) return tW > xW ? "Tài" : "Xỉu";
  return last20.at(-1) === "Tài" ? "Xỉu" : "Tài";
}

function shortPattern(hist) {
  if (hist.length < 6) return Math.random() < 0.5 ? "Tài" : "Xỉu";
  const last10 = hist.slice(-10).map(x => x.result);
  const pc = {};
  for (let i = 0; i <= last10.length - 4; i++) {
    const k = last10.slice(i, i + 4).join(",");
    pc[k] = (pc[k] || 0) + 1;
  }
  const top = Object.entries(pc).sort((a, b) => b[1] - a[1])[0];
  if (top && top[1] >= 3) {
    const end = last10.slice(-3).join(",");
    // nếu 3 cuối của pattern khớp 3 cuối hiện tại → dự đoán phần tử tiếp theo của pattern
    const parts = top[0].split(",");
    if (end === parts.slice(0, 3).join(",")) return parts[3];
  }
  return last10.at(-1) === "Tài" ? "Xỉu" : "Tài";
}

function meanDeviation(hist) {
  const last15 = hist.slice(-15).map(x => x.result);
  if (last15.length < 8) return Math.random() < 0.5 ? "Tài" : "Xỉu";
  const t = last15.filter(r => r === "Tài").length;
  const x = last15.length - t;
  if (Math.abs(t - x) / last15.length < 0.25) return last15.at(-1) === "Tài" ? "Xỉu" : "Tài";
  return t > x ? "Xỉu" : "Tài";
}

function recentSwitch(hist) {
  const last12 = hist.slice(-12).map(x => x.result);
  if (last12.length < 6) return Math.random() < 0.5 ? "Tài" : "Xỉu";
  const sw = last12.slice(1).reduce((c, r, i) => c + (r !== last12[i] ? 1 : 0), 0);
  return sw >= 7 ? (last12.at(-1) === "Tài" ? "Xỉu" : "Tài")
                 : (last12.at(-1) === "Tài" ? "Xỉu" : "Tài");
}

// ============= Mẫu cầu (n-gram lịch sử dài) =============
function mauCauPredict() {
  if (history.length < 8) return { pred: null, weight: 0, note: "Thiếu dài chuỗi" };
  const seq = history.map(h => toTX(h.result));
  const window = seq.slice(-5);
  const tries = [
    { n: 5, key: window.slice(-5).join("") },
    { n: 4, key: window.slice(-4).join("") },
    { n: 3, key: window.slice(-3).join("") },
  ];
  for (const t of tries) {
    const mem = patternMemory[t.n]?.[t.key];
    if (mem && (mem.T > 0 || mem.X > 0)) {
      const predTX = mem.T === mem.X ? (Math.random() < 0.5 ? "T" : "X") : (mem.T > mem.X ? "T" : "X");
      const conf = Math.max(mem.T, mem.X) / (mem.T + mem.X);
      return { pred: fromTX(predTX), weight: 0.25 + conf * 0.25, note: `Mẫu ${t.n}-gram (${t.key}→${predTX})` };
    }
  }
  return { pred: null, weight: 0.0, note: "Không khớp mẫu n-gram" };
}

// ============= Tự đánh giá hiệu năng để scale weight =============
function evaluatePerformance(modelName, lookback = 18) {
  const keys = Object.keys(modelPredictions).map(Number).sort((a, b) => a - b);
  if (keys.length < 3) return 1;
  const use = keys.slice(-lookback - 1);
  let correct = 0, total = 0;
  for (let i = 1; i < use.length; i++) {
    const prev = use[i - 1];
    const m = modelPredictions[prev];
    const actual = history.find(h => h.session === prev)?.result;
    if (!m || !actual) continue;
    if (m[modelName] === actual) correct++;
    total++;
  }
  if (!total) return 1;
  const ratio = 1 + (correct - total / 2) / (total / 2);
  return clamp(ratio, 0.6, 1.6);
}

function isBadPattern(hist) {
  const last20 = hist.slice(-20).map(x => x.result);
  const sw = last20.slice(1).reduce((c, r, i) => c + (r !== last20[i] ? 1 : 0), 0);
  const { streak } = detectStreak(hist);
  return sw >= 12 || streak >= 11;
}

// ============= Hợp nhất để ra dự đoán cuối =============
function finalPredict() {
  if (history.length === 0) {
    return { pred: Math.random() < 0.5 ? "Tài" : "Xỉu", conf: 0.5, explain: "Không có dữ liệu" };
  }
  const curSession = history.at(-1).session;

  // module outputs
  const tPred = trendWeighted(history);
  const sPred = shortPattern(history);
  const mPred = meanDeviation(history);
  const swPred = recentSwitch(history);
  const mk = markovTransition(history);
  const brProb = breakProbability(history);
  const pro = aiProAnalyzer(history);
  const mc = mauCauPredict();
  const mc15 = mauCau15Predict(); // MỚI – dùng chuỗi 15 ký tự

  // weights (có scale bằng hiệu năng)
  const perf = (name, lb = 18) => evaluatePerformance(name, lb);
  const weights = {
    trend:  0.16 * perf("trend"),
    short:  0.16 * perf("short"),
    mean:   0.18 * perf("mean"),
    switch: 0.14 * perf("switch"),
    markov: 0.10 * perf("markov"),
    proAI:  0.22 * perf("proAI"),
    mau:    mc.weight || 0.18,
    mau15:  0.24 * perf("mau15")
  };

  // accumulate scores
  let tai = 0, xiu = 0;
  const add = (p, w) => { if (p === "Tài") tai += w; else if (p === "Xỉu") xiu += w; };

  add(tPred, weights.trend);
  add(sPred, weights.short);
  add(mPred, weights.mean);
  add(swPred, weights.switch);
  add(mk.pT >= 0.5 ? "Tài" : "Xỉu", weights.markov);
  add(pro.pred, weights.proAI);
  if (mc.pred)  add(mc.pred, weights.mau);
  if (mc15.pred) add(mc15.pred, weights.mau15);

  // momentum gần 5 phiên
  const last5 = history.slice(-5).map(x => x.result);
  const mom = last5.filter(r => r === "Tài").length - last5.filter(r => r === "Xỉu").length;
  if (mom > 1) tai += 0.08; else if (mom < -1) xiu += 0.08;

  if (isBadPattern(history)) { tai *= 0.82; xiu *= 0.82; }

  // điều chỉnh thiên lệch 15 phiên
  const last15 = history.slice(-15).map(x => x.result);
  const t15 = last15.filter(r => r === "Tài").length;
  if (t15 >= 10) xiu += 0.12;
  else if (t15 <= 5) tai += 0.12;

  const pred = tai >= xiu ? "Tài" : "Xỉu";
  const margin = Math.abs(tai - xiu);
  // conf dựa trên margin + các module xác suất
  let conf = 0.55 + clamp(margin, 0, 0.9);
  // ảnh hưởng của mau_cau-15 (có conf riêng)
  conf = 0.6 * conf + 0.4 * (mc15.conf || 0.55);
  // ảnh hưởng bẻ cầu
  conf = pred === (history.at(-1).result === "Tài" ? "Xỉu" : "Tài")
    ? conf + brProb * 0.05
    : conf;
  conf = clamp(conf, 0.52, 0.98);

  // lưu cho tự chấm
  if (!modelPredictions[curSession]) modelPredictions[curSession] = {};
  modelPredictions[curSession] = {
    trend: tPred,
    short: sPred,
    mean: mPred,
    switch: swPred,
    markov: mk.pT >= 0.5 ? "Tài" : "Xỉu",
    proAI: pro.pred,
    mau: mc.pred || "None",
    mau15: mc15.pred || "None",
    final: pred
  };

  const explain = [
    `PRO(${pro.reason})`,
    `Markov pT=${r01(mk.pT)}`,
    `BreakProb=${r01(brProb)}`,
    `Mẫu_cầu=${mc.note}`,
    `Mau_cau15=${mc15.note}; conf15=${r01(mc15.conf)}`,
    `Điểm T=${r01(tai)} · X=${r01(xiu)} · margin=${r01(margin)}`
  ].join(" | ");

  return { pred, conf: r01(conf), explain };
}

// ============= Endpoint chính =============
app.get("/api/custom", async (_req, res) => {
  try {
    await loadSource();

    if (history.length === 0) {
      return res.status(500).json({ error: "Không có dữ liệu lịch sử" });
    }

    const latest = history.at(-1);
    const { session, dice, total, result } = latest;

    // Chuỗi Mau_cau 20 phiên gần nhất (giữ nguyên kiểu cũ)
    const Mau_cau = history.slice(-20).map(h => toTX(h.result)).join("");

    const { pred, conf, explain } = finalPredict();

    res.json({
      id: "Tele@idol_vannhat",
      Phien_truoc: session,
      Phien_sau: session + 1,
      Xuc_xac: dice,
      Tong: total,
      Ket_qua: result,
      Du_doan: pred,
      Do_tin_cay: conf,
      Giai_thich: explain,
      Mau_cau
    });
  } catch (err) {
    res.status(500).json({ error: "Lỗi xử lý", detail: err.message });
  }
});

// ============= Debug/Health =============
app.get("/", (_req, res) => res.send("Taixiu SIÊU VIP API ok. Use /api/custom"));
app.get("/debug/history", (_req, res) => res.json(history.slice(-80)));
app.get("/debug/pattern", (_req, res) => res.json(patternMemory));
app.get("/debug/predict_modules", (_req, res) => {
  const cur = history.at(-1)?.session;
  res.json({ session: cur, modules: modelPredictions[cur] || null });
});

app.listen(PORT, () => {
  console.log(`✅ API chạy: http://localhost:${PORT}/api/custom`);
});
