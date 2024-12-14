function detectPriceSignal(prices, previousState = {
    buySellSwitch: null,
    previousSBS: null,
    consecutiveHigherCloses: 0,
    consecutiveLowerCloses: 0,
    minSwing: 0,
    lastSignalPrice: null
}) {
    // Track consecutive closes
    let consecutiveHigherCloses = previousState.consecutiveHigherCloses;
    let consecutiveLowerCloses = previousState.consecutiveLowerCloses;
    
    // Check for consecutive closes
    if (prices.p0.close > prices.p1.close) {
        consecutiveHigherCloses++;
        consecutiveLowerCloses = 0;
    } else if (prices.p0.close < prices.p1.close) {
        consecutiveLowerCloses++;
        consecutiveHigherCloses = 0;
    } else {
        // Reset both on equal closes
        consecutiveHigherCloses = 0;
        consecutiveLowerCloses = 0;
    }

    // Determine if we have enough consecutive moves for a signal
    const hasSellSignal = consecutiveLowerCloses === 3;
    const hasBuySignal = consecutiveHigherCloses === 3;

    // Calculate pivot levels for signals
    let pivotHigh = null;
    let pivotLow = null;
    
    if (hasSellSignal) {
        // For sell signal, get the high from the first bar of the 3-bar series
        pivotHigh = prices.p2.high;
    }
    if (hasBuySignal) {
        // For buy signal, get the low from the first bar of the 3-bar series
        pivotLow = prices.p2.low;
    }

    // Check minimum swing requirement
    let isValidSwing = false;
    if (previousState.lastSignalPrice && (pivotHigh || pivotLow)) {
        const swingSize = Math.abs((pivotHigh || pivotLow) - previousState.lastSignalPrice);
        isValidSwing = swingSize >= previousState.minSwing;
    } else {
        isValidSwing = true; // First signal doesn't need swing check
    }

    return {
        consecutiveHigherCloses,
        consecutiveLowerCloses,
        minSwing: previousState.minSwing,
        lastSignalPrice: (hasSellSignal || hasBuySignal) ? prices.p0.close : previousState.lastSignalPrice,
        isNewSell: hasSellSignal && isValidSwing,
        isNewBuy: hasBuySignal && isValidSwing,
        pivotHigh: hasSellSignal && isValidSwing ? pivotHigh : null,
        pivotLow: hasBuySignal && isValidSwing ? pivotLow : null
    };
}

function criteriaCheck(backTestFrom, indicators, priceDataObj, priceDateArry, minSwing = 0) {
    const tradeSignals = [];
    let previousState = {
        consecutiveHigherCloses: 0,
        consecutiveLowerCloses: 0,
        minSwing: minSwing,
        lastSignalPrice: null
    };

    for (let i = 5000; i >= 0; i--) {
        // Need at least 3 bars of data for the signal
        if (i + 2 >= priceDateArry.length) continue;

        const prices = {
            p0: {
                close: parseFloat(priceDataObj[priceDateArry[i]]["4. close"]),
                high: parseFloat(priceDataObj[priceDateArry[i]]["2. high"]),
                low: parseFloat(priceDataObj[priceDateArry[i]]["3. low"])
            },
            p1: {
                close: parseFloat(priceDataObj[priceDateArry[i + 1]]["4. close"]),
                high: parseFloat(priceDataObj[priceDateArry[i + 1]]["2. high"]),
                low: parseFloat(priceDataObj[priceDateArry[i + 1]]["3. low"])
            },
            p2: {
                close: parseFloat(priceDataObj[priceDateArry[i + 2]]["4. close"]),
                high: parseFloat(priceDataObj[priceDateArry[i + 2]]["2. high"]),
                low: parseFloat(priceDataObj[priceDateArry[i + 2]]["3. low"])
            }
        };

        const signal = detectPriceSignal(prices, previousState);

        if (signal.isNewBuy || signal.isNewSell) {
            tradeSignals.push({
                date: priceDateArry[i],
                type: signal.isNewBuy ? 'BUY' : 'SELL',
                price: prices.p0.close,
                pivotLevel: signal.isNewBuy ? signal.pivotLow : signal.pivotHigh,
                consecutiveCloses: signal.isNewBuy ? 
                    signal.consecutiveHigherCloses : 
                    signal.consecutiveLowerCloses
            });
        }

        previousState = signal;
    }

    return tradeSignals;
}