/**
 * API Taixiu ULTIMATE VIP PRO v2 – by Tele@idol_vannhat
 * Node >= 18 (có fetch sẵn)
 * Endpoint: GET /api/predict
 * Trả về:
 * {
 * id: "Tele@idol_vannhat",
 * Phien_truoc, Phien_sau, Xuc_xac, Tong, Ket_qua, Du_doan, Do_tin_cay, Giai_thich, Mau_cau
 * }
 */

import express from "express";

const app = express();
const PORT = process.env.PORT || 3000;

// ============= Cấu hình nguồn dữ liệu =============
const SOURCE_API = "https://sunai.onrender.com/api/taixiu/history";

// ============= Bộ nhớ trong server =============
let history = []; // [{session, dice:[a,b,c], total, result}]
let patternMemory = {}; // n-gram 'T'/'X' -> tần suất tiếp theo
let modelPredictions = {}; // {session: {module: "Tài"/"Xỉu", ... , final}}
let predictionsHistory = []; // {session, du_doan, ket_qua, do_tin_cay, danh_gia}
let modelWeights = { markov: 0.15, proAI: 0.25, mau15: 0.35, dice: 0.25 }; // Trọng số khởi tạo
const MAX_HISTORY = 300; // Giảm để tối ưu bộ nhớ
const MAX_PREDICTIONS_HISTORY = 150;

// ============= Tiện ích chung =============
const r01 = (x) => Math.round(x * 100) / 100;
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
const last = (arr, k = 1) => arr.slice(-k);
const toTX = (res) => (res === "Tài" ? "T" : "X");
const fromTX = (c) => (c === "T" ? "Tài" : "Xỉu");

// ============= Nạp & chuẩn hoá dữ liệu =============
async function loadSource() {
  try {
    const resp = await fetch(SOURCE_API, { cache: "no-store" });
    const data = await resp.json();
    if (!Array.isArray(data) || data.length === 0) {
      throw new Error("API nguồn không trả mảng dữ liệu hợp lệ");
    }
    const norm = data
      .filter((x) => x && typeof x.session !== "undefined")
      .map((x) => ({
        session: Number(x.session),
        dice: Array.isArray(x.dice) ? x.dice.map(Number) : [0, 0, 0],
        total: Number(x.total ?? 0),
        result: x.result === "Tài" || x.result === "Xỉu" ? x.result : Number(x.total ?? 0) >= 11 ? "Tài" : "Xỉu",
      }))
      .sort((a, b) => a.session - b.session);

    const seen = new Set(history.map((h) => h.session));
    for (const row of norm) if (!seen.has(row.session)) history.push(row);
    if (history.length > MAX_HISTORY) history = history.slice(-MAX_HISTORY);

    rebuildPatternMemory();
    updateModelWeights(); // Cập nhật trọng số sau khi nạp dữ liệu
  } catch (err) {
    console.error("Lỗi loadSource:", err.message);
  }
}

function rebuildPatternMemory() {
  patternMemory = {};
  const seq = history.map((h) => toTX(h.result));
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

// ============= Tự động cập nhật trọng số (Logistic Regression đơn giản) =============
function updateModelWeights(lookback = 50) {
  const keys = Object.keys(modelPredictions).map(Number).sort((a, b) => a - b);
  if (keys.length < 10) return;

  const use = keys.slice(-lookback);
  const models = ["markov", "proAI", "mau15", "dice"];
  let correct = { markov: 0, proAI: 0, mau15: 0, dice: 0 };
  let total = 0;

  for (let i = 1; i < use.length; i++) {
    const prev = use[i - 1];
    const m = modelPredictions[prev];
    const actual = history.find((h) => h.session === prev)?.result;
    if (!m || !actual) continue;
    models.forEach((model) => {
      if (m[model] === actual) correct[model]++;
    });
    total++;
  }

  if (total === 0) return;
  const learningRate = 0.05;
  models.forEach((model) => {
    const accuracy = correct[model] / total;
    modelWeights[model] = clamp(modelWeights[model] + learningRate * (accuracy - 0.5), 0.1, 0.4);
  });

  // Chuẩn hóa trọng số
  const sum = Object.values(modelWeights).reduce((s, w) => s + w, 0);
  models.forEach((model) => {
    modelWeights[model] = r01(modelWeights[model] / sum);
  });
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
  const last20 = hist.slice(-20).map((x) => x.result);
  const switches = last20.slice(1).reduce((c, r, i) => c + (r !== last20[i] ? 1 : 0), 0);
  const taiCount = last20.filter((r) => r === "Tài").length;
  const pT = last20.length ? taiCount / last20.length : 0.5;
  const pX = 1 - pT;
  const entropy = -(pT ? pT * Math.log2(pT) : 0) - (pX ? pX * Math.log2(pX) : 0);

  let base = streak >= 9 ? 0.85 : streak >= 7 ? 0.75 : streak >= 5 ? 0.6 : streak >= 4 ? 0.45 : 0.35;
  base += switches / 45;
  base += entropy < 0.8 ? 0.12 : 0;

  return clamp(base, 0.1, 0.98);
}

function markovTransition(hist) {
  if (hist.length < 6) return { pT: 0.5, pX: 0.5 };
  const M = { Tài: { Tài: 0, Xỉu: 0 }, Xỉu: { Tài: 0, Xỉu: 0 } };
  for (let i = 1; i < hist.length; i++) M[hist[i - 1].result][hist[i].result]++;
  const cur = hist.at(-1).result;
  const a = M[cur].Tài + 1;
  const b = M[cur].Xỉu + 1;
  const pT = a / (a + b);
  return { pT, pX: 1 - pT };
}

function totalsStats(hist, k = 15) {
  const v = hist.slice(-k).map((x) => x.total || 0);
  const avg = v.reduce((s, x) => s + x, 0) / (v.length || 1);
  const varc = v.reduce((s, x) => s + (x - avg) ** 2, 0) / (v.length || 1);
  return { avg, varc };
}

// ============= Phân tích xúc xắc (Nâng cao) =============
function diceAnalyzer(hist, k = 15) {
  if (hist.length < k) return { pred: null, conf: 0.5, note: "Thiếu dữ liệu xúc xắc" };

  const recent = hist.slice(-k);
  const diceValues = recent.flatMap((h) => h.dice);
  const totalAvg = recent.reduce((s, h) => s + h.total, 0) / k;

  const freq = { 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0 };
  diceValues.forEach((d) => freq[d]++);
  const highFreq = (freq[5] + freq[6]) / (k * 3);
  const lowFreq = (freq[1] + freq[2]) / (k * 3);

  // Phát hiện cặp xúc xắc lặp lại
  const pairs = {};
  recent.forEach((h) => {
    const key = h.dice.sort().join(",");
    pairs[key] = (pairs[key] || 0) + 1;
  });
  const topPair = Object.entries(pairs).sort((a, b) => b[1] - a[1])[0];
  const pairBoost = topPair && topPair[1] >= 3 ? 0.1 : 0;

  let pT = 0.5;
  if (totalAvg > 11 || highFreq > 0.35) pT += 0.2 + pairBoost;
  if (totalAvg < 10 || lowFreq > 0.45) pT -= 0.2 + pairBoost;

  const pred = pT >= 0.5 ? "Tài" : "Xỉu";
  const conf = clamp(0.6 + Math.abs(pT - 0.5) * 0.95 + pairBoost, 0.55, 0.92);

  return {
    pred,
    conf,
    note: `Dice: avg=${r01(totalAvg)}, highFreq=${r01(highFreq)}, lowFreq=${r01(lowFreq)}, topPair=${topPair ? topPair[0] : "none"}`,
  };
}

// ============= Phân tích Mau_cau-15 (Siêu cải tiến) =============
function mauCau15Predict() {
  if (history.length < 16) return { pred: null, conf: 0.0, note: "Thiếu lịch sử" };

  const seq = history.map((h) => toTX(h.result));
  const MC = seq.slice(-15);
  const contexts = [
    { n: 5, key: seq.slice(-5).join("") },
    { n: 4, key: seq.slice(-4).join("") },
    { n: 3, key: seq.slice(-3).join("") },
  ];

  let pT = 0.5,
    used = null;
  for (const ctx of contexts) {
    const mem = patternMemory[ctx.n]?.[ctx.key];
    if (mem) {
      const aT = mem.T + 1;
      const aX = mem.X + 1;
      const post = aT / (aT + aX);
      pT = 0.65 * post + 0.35 * pT;
      used = `${ctx.n}-gram ${ctx.key}→T:${mem.T}/X:${mem.X}`;
    }
  }

  // Phân tích cấu trúc cầu
  let r = 1,
    lastC = MC[0],
    maxR = 1,
    runs = [];
  for (let i = 1; i < MC.length; i++) {
    if (MC[i] === lastC) {
      r++;
      maxR = Math.max(maxR, r);
    } else {
      runs.push({ type: lastC, len: r });
      r = 1;
      lastC = MC[i];
    }
  }
  runs.push({ type: lastC, len: r });

  const avgRunLen = runs.reduce((s, r) => s + r.len, 0) / (runs.length || 1);
  if (maxR >= 5 || (runs.at(-1).len >= avgRunLen + 1)) {
    const lastChar = MC.at(-1);
    pT = lastChar === "T" ? clamp(pT - 0.2, 0.05, 0.95) : clamp(pT + 0.2, 0.05, 0.95);
  }

  // Phát hiện mẫu bậc thang
  let stair = 0;
  const last8 = seq.slice(-8);
  if (last8.length >= 8) {
    for (let i = 2; i < last8.length; i++) {
      if (last8[i] === last8[i - 1] && last8[i - 2] !== last8[i - 1]) stair++;
    }
  }
  if (stair >= 2) {
    const lastChar = MC.at(-1);
    pT = lastChar === "T" ? clamp(pT - 0.1, 0.05, 0.95) : clamp(pT + 0.1, 0.05, 0.95);
  }

  const sw = MC.slice(1).reduce((c, ch, i) => c + (ch !== MC[i] ? 1 : 0), 0);
  if (sw >= 12) {
    const lastChar = MC.at(-1);
    pT = lastChar === "T" ? clamp(pT - 0.15, 0.05, 0.95) : clamp(pT + 0.15, 0.05, 0.95);
  }

  // Entropy động
  const windows = [10, 15, 20];
  let entropyAvg = 0;
  windows.forEach((w) => {
    const subSeq = seq.slice(-w);
    if (subSeq.length === w) {
      const cntT = subSeq.filter((c) => c === "T").length;
      const pT_w = cntT / w;
      const pX_w = 1 - pT_w;
      const H = -(pT_w ? pT_w * Math.log2(pT_w) : 0) - (pX_w ? pX_w * Math.log2(pX_w) : 0);
      entropyAvg += H / windows.length;
    }
  });

  const confBoost = entropyAvg < 0.8 ? 0.12 : 0;
  const pred = pT >= 0.5 ? "Tài" : "Xỉu";
  const conf = clamp(0.62 + Math.abs(pT - 0.5) * 0.95 + confBoost, 0.6, 0.95);

  return { pred, conf, note: used ? `Mau_cau15: ${used}, pT=${r01(pT)}, runs=${runs.length}, stair=${stair}` : `Mau_cau15: pT=${r01(pT)}` };
}

// ============= AI phân tích cầu PRO (Siêu cải tiến) =============
function aiProAnalyzer(hist) {
  if (hist.length < 7) return { pred: Math.random() < 0.5 ? "Tài" : "Xỉu", reason: "Thiếu dữ liệu" };

  const { streak, current } = detectStreak(hist);
  const bprob = breakProbability(hist);
  const { avg, varc } = totalsStats(hist, 15);
  const last8 = hist.slice(-8).map((x) => x.result);

  let stair = 0;
  if (last8.length >= 3) {
    for (let i = 2; i < last8.length; i++) {
      if (last8[i] === last8[i - 1] && last8[i - 2] !== last8[i - 1]) stair++;
    }
  }

  const sw8 = last8.slice(1).reduce((c, r, i) => c + (r !== last8[i] ? 1 : 0), 0);

  let scoreBreak = 0;
  scoreBreak += streak >= 7 ? 0.45 : streak >= 5 ? 0.3 : streak >= 4 ? 0.2 : 0.12;
  scoreBreak += bprob * 0.7;
  scoreBreak += varc > 10 ? 0.15 : 0;
  scoreBreak += stair >= 2 ? 0.1 : 0;
  scoreBreak += sw8 >= 6 ? 0.08 : 0;

  if (avg > 11.2) scoreBreak += current === "Tài" ? -0.06 : 0.1;
  if (avg < 9.8) scoreBreak += current === "Xỉu" ? -0.06 : 0.1;

  scoreBreak = clamp(scoreBreak, 0, 0.96);
  const pred = scoreBreak > 0.62 ? (current === "Tài" ? "Xỉu" : "Tài") : current;

  return {
    pred,
    reason: `Streak=${streak}, BreakProb=${r01(bprob)}, avg=${r01(avg)}, var=${r01(varc)}, stair=${stair}, sw8=${sw8}`,
  };
}

// ============= Hợp nhất dự đoán (Ultimate VIP PRO v2) =============
function finalPredict() {
  if (history.length === 0) {
    return { pred: Math.random() < 0.5 ? "Tài" : "Xỉu", conf: 0.5, explain: "Không có dữ liệu" };
  }
  const curSession = history.at(-1).session;

  // Module outputs
  const mk = markovTransition(history);
  const pro = aiProAnalyzer(history);
  const mc15 = mauCau15Predict();
  const dice = diceAnalyzer(history);

  // Khai báo biến `stair` ở đầu hàm
  let stair = 0;
  const last8 = history.slice(-8).map((x) => x.result);
  if (last8.length >= 3) {
    for (let i = 2; i < last8.length; i++) {
      if (last8[i] === last8[i - 1] && last8[i - 2] !== last8[i - 1]) stair++;
    }
  }

  // Trọng số động
  const weights = modelWeights;

  let tai = 0,
    xiu = 0;
  const add = (p, w) => {
    if (p === "Tài") tai += w;
    else if (p === "Xỉu") xiu += w;
  };

  add(mk.pT >= 0.5 ? "Tài" : "Xỉu", weights.markov);
  add(pro.pred, weights.proAI);
  if (mc15.pred) add(mc15.pred, weights.mau15);
  if (dice.pred) add(dice.pred, weights.dice);

  // Điều chỉnh mẫu cầu đặc biệt
  const last20 = history.slice(-20).map((x) => x.result);
  const sw = last20.slice(1).reduce((c, r, i) => c + (r !== last20[i] ? 1 : 0), 0);
  const { streak, current } = detectStreak(history);

  if (streak >= 5) {
    const breakProb = clamp(0.75 + (streak - 5) * 0.06, 0.75, 0.92);
    if (current === "Tài") xiu += breakProb * 0.35;
    else tai += breakProb * 0.35;
  }
  if (sw >= 12) {
    if (last20.at(-1) === "Tài") xiu += 0.25;
    else tai += 0.25;
  }

  const last15 = history.slice(-15).map((x) => x.result);
  const t15 = last15.filter((r) => r === "Tài").length;
  if (t15 >= 11) xiu += 0.2;
  else if (t15 <= 4) tai += 0.2;

  const pred = tai >= xiu ? "Tài" : "Xỉu";
  const margin = Math.abs(tai - xiu);
  let conf = 0.6 + clamp(margin, 0, 0.95);
  conf = 0.5 * conf + 0.3 * (mc15.conf || 0.6) + 0.2 * (dice.conf || 0.6);
  conf = clamp(conf, 0.6, 0.98);

  // Phân tích mẫu cầu
  const patternNote =
    sw >= 12
      ? "Cầu alternating mạnh"
      : streak >= 5
      ? `Cầu bệt (${streak} ${current})`
      : stair >= 2
      ? "Cầu bậc thang"
      : "Cầu ngẫu nhiên";

  // Lưu dự đoán
  if (!modelPredictions[curSession]) modelPredictions[curSession] = {};
  modelPredictions[curSession] = {
    markov: mk.pT >= 0.5 ? "Tài" : "Xỉu",
    proAI: pro.pred,
    mau15: mc15.pred || "None",
    dice: dice.pred || "None",
    final: pred,
  };

  const latestPrediction = {
    Phien_truoc: curSession,
    Du_doan: pred,
    Ket_qua: history.at(-1).result,
    Do_tin_cay: r01(conf),
    Danh_gia: pred === history.at(-1).result ? "ĐÚNG✅" : "SAI❌",
  };

  const seen = new Set(predictionsHistory.map((p) => p.Phien_truoc));
  if (!seen.has(curSession)) {
    predictionsHistory.push(latestPrediction);
    if (predictionsHistory.length > MAX_PREDICTIONS_HISTORY) {
      predictionsHistory = predictionsHistory.slice(-MAX_PREDICTIONS_HISTORY);
    }
  }

  const explain = [
    `PRO(${pro.reason})`,
    `Markov pT=${r01(mk.pT)}`,
    `BreakProb=${r01(breakProbability(history))}`,
    `Mau_cau15=${mc15.note}; conf15=${r01(mc15.conf)}`,
    `Dice=${dice.note}`,
    `Pattern=${patternNote}`,
    `Điểm T=${r01(tai)} · X=${r01(xiu)} · margin=${r01(margin)}`,
    `Weights=${JSON.stringify(modelWeights)}`,
  ].join(" | ");

  return { pred, conf: r01(conf), explain };
}

// ============= Tự động làm mới dữ liệu =============
setInterval(loadSource, 5 * 60 * 1000); // Làm mới mỗi 5 phút

// ============= Endpoint chính =============
app.get("/api/predict", async (_req, res) => {
  try {
    await loadSource();

    if (history.length === 0) {
      return res.status(500).json({ error: "Không có dữ liệu lịch sử" });
    }

    const latest = history.at(-1);
    const { session, dice, total, result } = latest;
    const Mau_cau = history.slice(-20).map((h) => toTX(h.result)).join("");
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
      Mau_cau,
    });
  } catch (err) {
    res.status(500).json({ error: "Lỗi xử lý", detail: err.message });
  }
});

// ============= Debug/Health =============
app.get("/", (_req, res) => res.send("Taixiu ULTIMATE VIP PRO v2 API ok. Use /api/predict"));
app.get("/debug/history", (_req, res) => res.json(history.slice(-80)));
app.get("/debug/pattern", (_req, res) => res.json(patternMemory));
app.get("/debug/predict_modules", (_req, res) => {
  const cur = history.at(-1)?.session;
  res.json({ session: cur, modules: modelPredictions[cur] || null });
});
app.get("/debug/lichsu-dudoan", (_req, res) => res.json(predictionsHistory));
app.get("/debug/weights", (_req, res) => res.json(modelWeights));

app.listen(PORT, () => {
  console.log(`✅ API chạy: http://localhost:${PORT}/api/predict`);
});
