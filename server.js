// server.js
const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const fetch = require('node-fetch');

const app = express();
const PORT = process.env.PORT || 3000;
const API_URL = "https://toilavinhmaycays23.onrender.com/vinhmaycay";

// Middleware
app.use(cors());
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));

// ==================== THUẬT TOÁN DỰ ĐOÁN ====================
// --- Dán toàn bộ các hàm JavaScript của bạn vào đây ---
// (Ví dụ đã được cung cấp ở câu trả lời trước, bạn chỉ cần dán vào)

function detectStreakAndBreak(history) {
    if (!history || history.length === 0) {
        return { streak: 0, currentResult: null, breakProb: 0 };
    }
    let streak = 1;
    const currentResult = history[history.length - 1].result;
    for (let i = history.length - 2; i >= 0; i--) {
        if (history[i].result === currentResult) streak++;
        else break;
    }
    const last15Results = history.slice(-15).map(item => item.result);
    if (!last15Results.length) {
        return { streak, currentResult, breakProb: 0 };
    }
    const switches = last15Results.slice(1).reduce((count, result, index) => {
        return count + (result !== last15Results[index] ? 1 : 0);
    }, 0);
    const taiCount = last15Results.filter(result => result === 'Tài').length;
    const xiuCount = last15Results.length - taiCount;
    const imbalance = Math.abs(taiCount - xiuCount) / last15Results.length;
    let breakProb = 0;
    if (streak >= 8) {
        breakProb = Math.min(0.6 + switches / 15 + imbalance * 0.15, 0.9);
    } else if (streak >= 5) {
        breakProb = Math.min(0.35 + switches / 10 + imbalance * 0.25, 0.85);
    } else if (streak >= 3 && switches >= 7) {
        breakProb = 0.3;
    }
    return { streak, currentResult, breakProb };
}

let modelPredictions = {};

function evaluateModelPerformance(history, modelName, lookback = 10) {
    if (!modelPredictions[modelName] || history.length < 2) return 1;
    lookback = Math.min(lookback, history.length - 1);
    let correctPredictions = 0;
    for (let i = 0; i < lookback; i++) {
        const sessionId = history[history.length - (i + 2)].session;
        const prediction = modelPredictions[modelName][sessionId] || 0;
        const actualResult = history[history.length - (i + 1)].result;
        if ((prediction === 1 && actualResult === 'Tài') ||
            (prediction === 2 && actualResult === 'Xỉu')) {
            correctPredictions++;
        }
    }
    const performanceRatio = lookback > 0 ?
        1 + (correctPredictions - lookback / 2) / (lookback / 2) : 1;
    return Math.max(0.5, Math.min(1.5, performanceRatio));
}

function smartBridgeBreak(history) {
    if (!history || history.length < 3) {
        return { prediction: 0, breakProb: 0, reason: 'Không đủ dữ liệu để bẻ cầu' };
    }
    const { streak, currentResult, breakProb } = detectStreakAndBreak(history);
    const last20Results = history.slice(-20).map(item => item.result);
    const last20Scores = history.slice(-20).map(item => item.score || 0);
    let finalBreakProb = breakProb;
    let reason = '';
    const avgScore = last20Scores.reduce((sum, score) => sum + score, 0) / (last20Scores.length || 1);
    const scoreDeviation = last20Scores.reduce((sum, score) => sum + Math.abs(score - avgScore), 0) / (last20Scores.length || 1);
    const last5Results = last20Results.slice(-5);
    const patternCounts = {};
    for (let i = 0; i <= last20Results.length - 3; i++) {
        const pattern = last20Results.slice(i, i + 3).join(',');
        patternCounts[pattern] = (patternCounts[pattern] || 0) + 1;
    }
    const mostCommonPattern = Object.entries(patternCounts).sort((a, b) => b[1] - a[1])[0];
    const hasRepeatingPattern = mostCommonPattern && mostCommonPattern[1] >= 3;
    if (streak >= 6) {
        finalBreakProb = Math.min(finalBreakProb + 0.15, 0.9);
        reason = `[Bẻ Cầu] Chuỗi ${streak} ${currentResult} dài, khả năng bẻ cầu cao`;
    } else if (streak >= 4 && scoreDeviation > 3) {
        finalBreakProb = Math.min(finalBreakProb + 0.1, 0.85);
        reason = `[Bẻ Cầu] Biến động điểm số lớn (${scoreDeviation.toFixed(1)}), khả năng bẻ cầu tăng`;
    } else if (hasRepeatingPattern && last5Results.every(result => result === currentResult)) {
        finalBreakProb = Math.min(finalBreakProb + 0.05, 0.8);
        reason = `[Bẻ Cầu] Phát hiện mẫu lặp ${mostCommonPattern[0]}, có khả năng bẻ cầu`;
    } else {
        finalBreakProb = Math.max(finalBreakProb - 0.15, 0.15);
        reason = '[Bẻ Cầu] Không phát hiện mẫu bẻ cầu mạnh, tiếp tục theo cầu';
    }
    let prediction = finalBreakProb > 0.65 ?
        (currentResult === 'Tài' ? 2 : 1) :
        (currentResult === 'Tài' ? 1 : 2);
    return { prediction, breakProb: finalBreakProb, reason };
}

function trendAndProb(history) {
    if (!history || history.length < 3) return 0;
    const { streak, currentResult, breakProb } = detectStreakAndBreak(history);
    if (streak >= 5) {
        if (breakProb > 0.75) return currentResult === 'Tài' ? 2 : 1;
        return currentResult === 'Tài' ? 1 : 2;
    }
    const last15Results = history.slice(-15).map(item => item.result);
    if (!last15Results.length) return 0;
    const weightedResults = last15Results.map((result, index) => Math.pow(1.2, index));
    const taiWeight = weightedResults.reduce((sum, weight, i) => sum + (last15Results[i] === 'Tài' ? weight : 0), 0);
    const xiuWeight = weightedResults.reduce((sum, weight, i) => sum + (last15Results[i] === 'Xỉu' ? weight : 0), 0);
    const totalWeight = taiWeight + xiuWeight;
    const last10Results = last15Results.slice(-10);
    const patterns = [];
    if (last10Results.length >= 4) {
        for (let i = 0; i <= last10Results.length - 4; i++) {
            patterns.push(last10Results.slice(i, i + 4).join(','));
        }
    }
    const patternCounts = patterns.reduce((counts, pattern) => {
        counts[pattern] = (counts[pattern] || 0) + 1;
        return counts;
    }, {});
    const mostCommonPattern = Object.entries(patternCounts).sort((a, b) => b[1] - a[1])[0];
    if (mostCommonPattern && mostCommonPattern[1] >= 3) {
        const patternParts = mostCommonPattern[0].split(',');
        return patternParts[patternParts.length - 1] !== last10Results[last10Results.length - 1] ? 1 : 2;
    }
    if (totalWeight > 0 && Math.abs(taiWeight - xiuWeight) / totalWeight >= 0.25) {
        return taiWeight > xiuWeight ? 2 : 1;
    }
    return last15Results[last15Results.length - 1] === 'Xỉu' ? 1 : 2;
}

function shortPattern(history) {
    if (!history || history.length < 3) return 0;
    const { streak, currentResult, breakProb } = detectStreakAndBreak(history);
    if (streak >= 4) {
        if (breakProb > 0.75) return currentResult === 'Tài' ? 2 : 1;
        return currentResult === 'Tài' ? 1 : 2;
    }
    const last8Results = history.slice(-8).map(item => item.result);
    if (!last8Results.length) return 0;
    const patterns = [];
    if (last8Results.length >= 3) {
        for (let i = 0; i <= last8Results.length - 3; i++) {
            patterns.push(last8Results.slice(i, i + 3).join(','));
        }
    }
    const patternCounts = patterns.reduce((counts, pattern) => {
        counts[pattern] = (counts[pattern] || 0) + 1;
        return counts;
    }, {});
    const mostCommonPattern = Object.entries(patternCounts).sort((a, b) => b[1] - a[1])[0];
    if (mostCommonPattern && mostCommonPattern[1] >= 2) {
        const patternParts = mostCommonPattern[0].split(',');
        return patternParts[patternParts.length - 1] !== last8Results[last8Results.length - 1] ? 1 : 2;
    }
    return last8Results[last8Results.length - 1] === 'Xỉu' ? 1 : 2;
}

function meanDeviation(history) {
    if (!history || history.length < 3) return 0;
    const { streak, currentResult, breakProb } = detectStreakAndBreak(history);
    if (streak >= 4) {
        if (breakProb > 0.75) return currentResult === 'Tài' ? 2 : 1;
        return currentResult === 'Tài' ? 1 : 2;
    }
    const last12Results = history.slice(-12).map(item => item.result);
    if (!last12Results.length) return 0;
    const taiCount = last12Results.filter(result => result === 'Tài').length;
    const xiuCount = last12Results.length - taiCount;
    const imbalance = Math.abs(taiCount - xiuCount) / last12Results.length;
    if (imbalance < 0.35) {
        return last12Results[last12Results.length - 1] === 'Xỉu' ? 1 : 2;
    }
    return xiuCount > taiCount ? 1 : 2;
}

function recentSwitch(history) {
    if (!history || history.length < 3) return 0;
    const { streak, currentResult, breakProb } = detectStreakAndBreak(history);
    if (streak >= 4) {
        if (breakProb > 0.75) return currentResult === 'Tài' ? 2 : 1;
        return currentResult === 'Tài' ? 1 : 2;
    }
    const last10Results = history.slice(-10).map(item => item.result);
    if (!last10Results.length) return 0;
    const switches = last10Results.slice(1).reduce((count, result, index) => {
        return count + (result !== last10Results[index] ? 1 : 0);
    }, 0);
    return switches >= 6 ?
        (last10Results[last10Results.length - 1] === 'Xỉu' ? 1 : 2) :
        (last10Results[last10Results.length - 1] === 'Xỉu' ? 1 : 2);
}

function isBadPattern(history) {
    if (!history || history.length < 3) return false;
    const last15Results = history.slice(-15).map(item => item.result);
    if (!last15Results.length) return false;
    const switches = last15Results.slice(1).reduce((count, result, index) => {
        return count + (result !== last15Results[index] ? 1 : 0);
    }, 0);
    const { streak } = detectStreakAndBreak(history);
    return switches >= 9 || streak >= 10;
}

function aiHtddLogic(history) {
    if (!history || history.length < 3) {
        const randomPred = Math.random() < 0.5 ? 'Tài' : 'Xỉu';
        return { prediction: randomPred, reason: 'Không đủ lịch sử, dự đoán ngẫu nhiên', source: 'AI HTDD' };
    }
    const last5Results = history.slice(-5).map(item => item.result);
    const last5Scores = history.slice(-5).map(item => item.score || 0);
    const taiCount = last5Results.filter(result => result === 'Tài').length;
    const xiuCount = last5Results.filter(result => result === 'Xỉu').length;
    if (history.length >= 3) {
        const last3Results = history.slice(-3).map(item => item.result);
        if (last3Results.join(',') === 'Tài,Xỉu,Tài') {
            return { prediction: 'Xỉu', reason: 'Phát hiện mẫu 1T1X → nên đánh Xỉu', source: 'AI HTDD' };
        } else if (last3Results.join(',') === 'Xỉu,Tài,Xỉu') {
            return { prediction: 'Tài', reason: 'Phát hiện mẫu 1X1T → nên đánh Tài', source: 'AI HTDD' };
        }
    }
    if (history.length >= 4) {
        const last4Results = history.slice(-4).map(item => item.result);
        if (last4Results.join(',') === 'Tài,Tài,Xỉu,Xỉu') {
            return { prediction: 'Tài', reason: 'Phát hiện mẫu 2T2X → nên đánh Tài', source: 'AI HTDD' };
        } else if (last4Results.join(',') === 'Xỉu,Xỉu,Tài,Tài') {
            return { prediction: 'Xỉu', reason: 'Phát hiện mẫu 2X2T → nên đánh Xỉu', source: 'AI HTDD' };
        }
    }
    if (history.length >= 9 && history.slice(-6).every(item => item.result === 'Tài')) {
        return { prediction: 'Xỉu', reason: 'Chuỗi Tài quá dài (6 lần) → dự đoán Xỉu', source: 'AI HTDD' };
    } else if (history.length >= 9 && history.slice(-6).every(item => item.result === 'Xỉu')) {
        return { prediction: 'Tài', reason: 'Chuỗi Xỉu quá dài (6 lần) → dự đoán Tài', source: 'AI HTDD' };
    }
    const avgScore = last5Scores.reduce((sum, score) => sum + score, 0) / (last5Scores.length || 1);
    if (avgScore > 10) {
        return { prediction: 'Tài', reason: `Điểm trung bình cao (${avgScore.toFixed(1)}) → dự đoán Tài`, source: 'AI HTDD' };
    } else if (avgScore < 8) {
        return { prediction: 'Xỉu', reason: `Điểm trung bình thấp (${avgScore.toFixed(1)}) → dự đoán Xỉu`, source: 'AI HTDD' };
    }
    if (taiCount > xiuCount + 1) {
        return { prediction: 'Xỉu', reason: `Tài chiếm đa số (${taiCount}/${last5Results.length}) → dự đoán Xỉu`, source: 'AI HTDD' };
    } else if (xiuCount > taiCount + 1) {
        return { prediction: 'Tài', reason: `Xỉu chiếm đa số (${xiuCount}/${last5Results.length}) → dự đoán Tài`, source: 'AI HTDD' };
    } else {
        const totalTai = history.filter(item => item.result === 'Tài').length;
        const totalXiu = history.filter(item => item.result === 'Xỉu').length;
        if (totalTai > totalXiu + 2) {
            return { prediction: 'Xỉu', reason: 'Tổng thể Tài nhiều hơn → dự đoán Xỉu', source: 'AI HTDD' };
        } else if (totalXiu > totalTai + 2) {
            return { prediction: 'Tài', reason: 'Tổng thể Xỉu nhiều hơn → dự đoán Tài', source: 'AI HTDD' };
        } else {
            const randomPred = Math.random() < 0.5 ? 'Tài' : 'Xỉu';
            return { prediction: randomPred, reason: 'Cân bằng, dự đoán ngẫu nhiên', source: 'AI HTDD' };
        }
    }
}

function generatePrediction(history, predictions) {
    modelPredictions = predictions || {};
    if (!history || history.length === 0) {
        const randomPred = Math.random() < 0.5 ? 'Tài' : 'Xỉu';
        return { prediction: randomPred, reason: 'Không đủ lịch sử', scores: { taiScore: 0, xiuScore: 0 } };
    }
    if (!modelPredictions.trend) {
        modelPredictions = { trend: {}, short: {}, mean: {}, switch: {}, bridge: {} };
    }
    const currentSession = history[history.length - 1].session;
    const trendPred = history.length < 5 ? (history[history.length - 1].result === 'Tài' ? 2 : 1) : trendAndProb(history);
    const shortPred = history.length < 5 ? (history[history.length - 1].result === 'Tài' ? 2 : 1) : shortPattern(history);
    const meanPred = history.length < 5 ? (history[history.length - 1].result === 'Tài' ? 2 : 1) : meanDeviation(history);
    const switchPred = history.length < 5 ? (history[history.length - 1].result === 'Tài' ? 2 : 1) : recentSwitch(history);
    const bridgePred = history.length < 5 ? { prediction: history[history.length - 1].result === 'Tài' ? 2 : 1, breakProb: 0, reason: 'Lịch sử ngắn' } : smartBridgeBreak(history);
    const aiPred = aiHtddLogic(history);
    modelPredictions.trend[currentSession] = trendPred;
    modelPredictions.short[currentSession] = shortPred;
    modelPredictions.mean[currentSession] = meanPred;
    modelPredictions.switch[currentSession] = switchPred;
    modelPredictions.bridge[currentSession] = bridgePred.prediction;
    const modelPerformance = {
        trend: evaluateModelPerformance(history, 'trend'),
        short: evaluateModelPerformance(history, 'short'),
        mean: evaluateModelPerformance(history, 'mean'),
        switch: evaluateModelPerformance(history, 'switch'),
        bridge: evaluateModelPerformance(history, 'bridge')
    };
    const modelWeights = {
        trend: 0.2 * modelPerformance.trend,
        short: 0.2 * modelPerformance.short,
        mean: 0.25 * modelPerformance.mean,
        switch: 0.2 * modelPerformance.switch,
        bridge: 0.15 * modelPerformance.bridge,
        aihtdd: 0.2
    };
    let taiScore = 0;
    let xiuScore = 0;
    if (trendPred === 1) taiScore += modelWeights.trend;
    else if (trendPred === 2) xiuScore += modelWeights.trend;
    if (shortPred === 1) taiScore += modelWeights.short;
    else if (shortPred === 2) xiuScore += modelWeights.short;
    if (meanPred === 1) taiScore += modelWeights.mean;
    else if (meanPred === 2) xiuScore += modelWeights.mean;
    if (switchPred === 1) taiScore += modelWeights.switch;
    else if (switchPred === 2) xiuScore += modelWeights.switch;
    if (bridgePred.prediction === 1) taiScore += modelWeights.bridge;
    else if (bridgePred.prediction === 2) xiuScore += modelWeights.bridge;
    if (aiPred.prediction === 'Tài') taiScore += modelWeights.aihtdd;
    else xiuScore += modelWeights.aihtdd;
    if (isBadPattern(history)) {
        taiScore *= 0.8;
        xiuScore *= 0.8;
    }
    const last10Results = history.slice(-10).map(item => item.result);
    const last10TaiCount = last10Results.filter(result => result === 'Tài').length;
    if (last10TaiCount >= 7) {
        xiuScore += 0.15;
    } else if (last10TaiCount <= 3) {
        taiScore += 0.15;
    }
    if (bridgePred.breakProb > 0.65) {
        if (bridgePred.prediction === 1) taiScore += 0.2;
        else xiuScore += 0.2;
    }
    const finalPrediction = taiScore > xiuScore ? 'Tài' : 'Xỉu';
    return { prediction: finalPrediction, reason: `${aiPred.reason} | ${bridgePred.reason}`, scores: { taiScore, xiuScore } };
}

// ==================== ENDPOINT API ====================

app.get('/api/predict', async (req, res) => {
    try {
        const response = await fetch(API_URL);
        if (!response.ok) {
            throw new Error('Failed to fetch data from game API');
        }
        const gameData = await response.json();
        
        // Tạo dữ liệu lịch sử giả định hoặc lấy từ một nguồn tin cậy
        const historyData = [
            { session: gameData.Phien - 5, result: "Tài", score: 12 },
            { session: gameData.Phien - 4, result: "Xỉu", score: 9 },
            { session: gameData.Phien - 3, result: "Tài", score: 10 },
            { session: gameData.Phien - 2, result: "Tài", score: 11 },
            { session: gameData.Phien - 1, result: "Xỉu", score: 8 }
        ];

        // Lấy kết quả cuối cùng từ API và thêm vào history
        const lastResult = (gameData.Xuc_xac_1 + gameData.Xuc_xac_2 + gameData.Xuc_xac_3) >= 11 ? 'Tài' : 'Xỉu';
        historyData.push({ session: gameData.Phien, result: lastResult, score: (gameData.Xuc_xac_1 + gameData.Xuc_xac_2 + gameData.Xuc_xac_3) });
        
        // Gọi hàm dự đoán của bạn
        const predictionResult = generatePrediction(historyData, {});

        const lastSession = historyData[historyData.length - 1];
        const phien_truoc = lastSession ? lastSession.session : null;
        const ket_qua = lastSession ? lastSession.result : null;
        const phien_sau = phien_truoc ? phien_truoc + 1 : null;
        const do_tin_cay = (predictionResult.scores.taiScore + predictionResult.scores.xiuScore).toFixed(2);
        const giai_thich = predictionResult.reason;
        
        res.json({
            Phien_truoc: phien_truoc,
            Ket_qua: ket_qua,
            Phien_sau: phien_sau,
            Du_doan: predictionResult.prediction,
            Do_tin_cay: do_tin_cay,
            Giai_thich: giai_thich,
            id: "Tele@Idol_vannhat"
        });

    } catch (error) {
        console.error('Error in API route:', error);
        res.status(500).json({ error: 'Failed to generate prediction', details: error.message });
    }
});

app.get('/', (req, res) => {
    res.send('API is running. Use GET /api/predict to get a prediction.');
});

// Khởi động server
app.listen(PORT, () => {
    console.log(`API server listening on port ${PORT}`);
});
