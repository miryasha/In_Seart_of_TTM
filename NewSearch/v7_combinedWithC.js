function detectPriceSignal(prices, previousState = {
    buySellSwitch: null,
    previousHigh: null,
    previousLow: null,
    n: 8 // lookback period
}) {
    // Exact TTMScalperStyle logic implementation
    const highestHigh = Math.max(...Array(previousState.n).fill().map((_, i) => 
        prices[`p${i}`]?.high || 0));
    
    const lowestLow = Math.min(...Array(previousState.n).fill().map((_, i) => 
        prices[`p${i}`]?.low || Infinity));
    
    const isHighestBar = prices.p0.high === highestHigh;
    const isLowestBar = prices.p0.low === lowestLow;
    
    // Update previous levels only when new highs/lows are made
    const currentHigh = isHighestBar ? highestHigh : previousState.previousHigh;
    const currentLow = isLowestBar ? lowestLow : previousState.previousLow;
    
    // Calculate signals based on new highs/lows
    const buySellSwitch = isHighestBar ? 0 : (isLowestBar ? 1 : previousState.buySellSwitch ?? 0);

    return {
        buySellSwitch,
        previousHigh: currentHigh,
        previousLow: currentLow,
        n: previousState.n,
        isNewSell: isLowestBar,
        isNewBuy: isHighestBar,
        pivotHigh: isHighestBar ? highestHigh : null,
        pivotLow: isLowestBar ? lowestLow : null
    };
}

function criteriaCheck(backTestFrom, indicators, priceDataObj, priceDateArry) {
    const tradeSignals = [];
    let previousState = {
        buySellSwitch: 0,
        previousHigh: null,
        previousLow: null,
        n: 8
    };

    for (let i = 5000; i >= 0; i--) {
        // Build price object for last n bars
        const prices = {};
        for (let j = 0; j < previousState.n; j++) {
            if (i + j >= priceDateArry.length) continue;
            
            prices[`p${j}`] = {
                close: parseFloat(priceDataObj[priceDateArry[i + j]]["4. close"]),
                high: parseFloat(priceDataObj[priceDateArry[i + j]]["2. high"]),
                low: parseFloat(priceDataObj[priceDateArry[i + j]]["3. low"])
            };
        }

        const signal = detectPriceSignal(prices, previousState);

        if (signal.isNewBuy || signal.isNewSell) {
            tradeSignals.push({
                date: priceDateArry[i],
                type: signal.isNewBuy ? 'BUY' : 'SELL',
                price: prices.p0.close,
                pivotHigh: signal.pivotHigh,
                pivotLow: signal.pivotLow,
                level: signal.isNewBuy ? signal.previousHigh : signal.previousLow
            });
        }

        previousState = signal;
    }

    return tradeSignals;
}