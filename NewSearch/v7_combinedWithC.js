function detectPriceSignal(prices, previousState = {
    buySellSwitch: null,
    previousHigh: null,
    previousLow: null,
    n: 8 // lookback period
}) {
    // Get last n bars of data
    const lastNBars = Array(previousState.n).fill().map((_, i) => prices[`p${i}`]);
    
    // Find highest high and lowest low in lookback period
    const highest = Math.max(...lastNBars.map(bar => bar.high));
    const lowest = Math.min(...lastNBars.map(bar => bar.low));
    
    // Check if current bar makes new high/low
    const isNewHigh = prices.p0.high === highest;
    const isNewLow = prices.p0.low === lowest;
    
    // Update previous levels
    let currentHigh = isNewHigh ? highest : previousState.previousHigh;
    let currentLow = isNewLow ? lowest : previousState.previousLow;
    
    // Generate signals
    const isSellSignal = prices.p0.low < currentLow;
    const isBuySignal = prices.p0.high > currentHigh;
    
    // Calculate buySellSwitch (0 for buy, 1 for sell)
    const buySellSwitch = isSellSignal ? 1 : (isBuySignal ? 0 : previousState.buySellSwitch ?? 0);

    return {
        buySellSwitch,
        previousHigh: currentHigh,
        previousLow: currentLow,
        n: previousState.n,
        isNewSell: isSellSignal,
        isNewBuy: isBuySignal
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
                level: signal.isNewBuy ? signal.previousHigh : signal.previousLow
            });
        }

        previousState = signal;
    }

    return tradeSignals;
}