const calculateADX = (prices, period = 14) => {
    const trueRanges = [];
    const dmPlus = [];
    const dmMinus = [];
    
    // Calculate True Range and Directional Movement
    for (let i = 1; i < prices.length; i++) {
        const high = prices[i].high;
        const low = prices[i].low;
        const prevHigh = prices[i - 1].high;
        const prevLow = prices[i - 1].low;
        const prevClose = prices[i - 1].close;
        
        // True Range
        const tr = Math.max(
            high - low,
            Math.abs(high - prevClose),
            Math.abs(low - prevClose)
        );
        trueRanges.push(tr);
        
        // Directional Movement
        const upMove = high - prevHigh;
        const downMove = prevLow - low;
        
        if (upMove > downMove && upMove > 0) {
            dmPlus.push(upMove);
            dmMinus.push(0);
        } else if (downMove > upMove && downMove > 0) {
            dmPlus.push(0);
            dmMinus.push(downMove);
        } else {
            dmPlus.push(0);
            dmMinus.push(0);
        }
    }
    
    // Calculate smoothed averages
    const smoothedTR = trueRanges.reduce((a, b) => a + b) / period;
    const smoothedDMPlus = dmPlus.reduce((a, b) => a + b) / period;
    const smoothedDMMinus = dmMinus.reduce((a, b) => a + b) / period;
    
    // Calculate DI+ and DI-
    const diPlus = (smoothedDMPlus / smoothedTR) * 100;
    const diMinus = (smoothedDMMinus / smoothedTR) * 100;
    
    // Calculate DX and ADX
    const dx = Math.abs(diPlus - diMinus) / (diPlus + diMinus) * 100;
    return dx;
};

const calculateEMA = (prices, period) => {
    const multiplier = 2 / (period + 1);
    let ema = prices[0].close;
    
    for (let i = 1; i < prices.length; i++) {
        ema = (prices[i].close - ema) * multiplier + ema;
    }
    
    return ema;
};

const calculateSMA = (prices, period) => {
    const sum = prices.slice(0, period).reduce((acc, price) => acc + price.close, 0);
    return sum / period;
};

const calculatePSAR = (prices, af = 0.02, maxAf = 0.2) => {
    let psar = prices[0].low;
    let ep = prices[0].high;
    let isUpTrend = true;
    let currentAf = af;
    
    for (let i = 1; i < prices.length; i++) {
        const high = prices[i].high;
        const low = prices[i].low;
        
        if (isUpTrend) {
            psar = psar + currentAf * (ep - psar);
            if (high > ep) {
                ep = high;
                currentAf = Math.min(currentAf + af, maxAf);
            }
            if (low < psar) {
                isUpTrend = false;
                psar = ep;
                ep = low;
                currentAf = af;
            }
        } else {
            psar = psar - currentAf * (psar - ep);
            if (low < ep) {
                ep = low;
                currentAf = Math.min(currentAf + af, maxAf);
            }
            if (high > psar) {
                isUpTrend = true;
                psar = ep;
                ep = high;
                currentAf = af;
            }
        }
    }
    
    return { psar, isUpTrend };
};

const calculateATR = (prices, period = 14) => {
    const trueRanges = [];
    
    for (let i = 1; i < prices.length; i++) {
        const tr = Math.max(
            prices[i].high - prices[i].low,
            Math.abs(prices[i].high - prices[i - 1].close),
            Math.abs(prices[i].low - prices[i - 1].close)
        );
        trueRanges.push(tr);
    }
    
    return trueRanges.reduce((a, b) => a + b) / period;
};

const TTMScalper = {
    analyze: function(prices, options = {}) {
        const {
            adxPeriod = 14,
            maType = 'EMA',
            maPeriod = 20,
            atrPeriod = 14,
            atrMultiplier = 2,
            useAdxFilter = true,
            useMaFilter = true,
            usePsarFilter = true,
            profitTarget = 'ATR', // or 'PERCENT'
            stopLossType = 'ATR', // or 'PSAR'
            profitTargetPercent = 1,
            minimumAdx = 25
        } = options;

        const signals = [];
        const lookback = Math.max(adxPeriod, maPeriod, atrPeriod);
        
        for (let i = lookback; i < prices.length; i++) {
            const priceWindow = prices.slice(i - lookback, i + 1);
            const currentPrice = prices[i].close;
            
            // Calculate indicators
            const adx = calculateADX(priceWindow, adxPeriod);
            const ma = maType === 'EMA' ? 
                calculateEMA(priceWindow, maPeriod) : 
                calculateSMA(priceWindow, maPeriod);
            const psar = calculatePSAR(priceWindow);
            const atr = calculateATR(priceWindow, atrPeriod);
            
            // Check trend filters
            const adxFilter = !useAdxFilter || adx > minimumAdx;
            const maFilter = !useMaFilter || (currentPrice > ma);
            const psarFilter = !usePsarFilter || psar.isUpTrend;
            
            // Generate signals
            if (adxFilter && maFilter && psarFilter) {
                const stopLoss = stopLossType === 'ATR' ? 
                    currentPrice - (atr * atrMultiplier) :
                    psar.psar;
                
                const takeProfit = profitTarget === 'ATR' ?
                    currentPrice + (atr * atrMultiplier) :
                    currentPrice * (1 + profitTargetPercent / 100);
                
                signals.push({
                    date: prices[i].date,
                    price: currentPrice,
                    type: 'BUY',
                    stopLoss,
                    takeProfit,
                    indicators: {
                        adx,
                        ma,
                        psar: psar.psar,
                        atr
                    }
                });
            }
        }
        
        return signals;
    }
};

module.exports = TTMScalper;




// const options = {
//     adxPeriod: 14,
//     maType: 'EMA', // or 'SMA'
//     maPeriod: 20,
//     atrPeriod: 14,
//     atrMultiplier: 2,
//     useAdxFilter: true,
//     useMaFilter: true,
//     usePsarFilter: true,
//     profitTarget: 'ATR', // or 'PERCENT'
//     stopLossType: 'ATR', // or 'PSAR'
//     profitTargetPercent: 1,
//     minimumAdx: 25
// };


const signals = TTMScalper.analyze(prices, options);