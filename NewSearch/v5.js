function detectPriceSignal(prices, previousState = {
    buySellSwitch: null,
    previousSBS: null,
    previousClrS: null,
    triggerBarHigh: null,
    triggerBarLow: null,
    consecutiveHigherCloses: 0,
    consecutiveLowerCloses: 0,
    minSwing: 0
}) {
    // Check for trigger bar conditions
    const higherLow = prices.p0.low > prices.p1.low;
    const lowerHigh = prices.p0.high < prices.p1.high;
    
    // Track consecutive closes
    let consecutiveHigherCloses = previousState.consecutiveHigherCloses;
    let consecutiveLowerCloses = previousState.consecutiveLowerCloses;
    
    if (prices.p0.close > prices.p1.close) {
        consecutiveHigherCloses++;
        consecutiveLowerCloses = 0;
    } else if (prices.p0.close < prices.p1.close) {
        consecutiveLowerCloses++;
        consecutiveHigherCloses = 0;
    }

    // Determine trigger conditions
    const triggerSell = lowerHigh && 
                       consecutiveLowerCloses >= 3 && 
                       prices.p0.close < previousState.triggerBarLow;

    const triggerBuy = higherLow && 
                      consecutiveHigherCloses >= 3 && 
                      prices.p0.close > previousState.triggerBarHigh;

    // Update trigger bars
    let triggerBarHigh = previousState.triggerBarHigh;
    let triggerBarLow = previousState.triggerBarLow;
    
    if (higherLow) {
        triggerBarHigh = prices.p0.high;
    }
    if (lowerHigh) {
        triggerBarLow = prices.p0.low;
    }

    // Calculate pivot levels (SBS)
    let SBS;
    if (triggerSell && previousState.buySellSwitch === 0) {
        SBS = Math.max(prices.p0.high, prices.p1.high, prices.p2.high); // Pivot High
    } else if (triggerBuy && previousState.buySellSwitch === 1) {
        SBS = Math.min(prices.p0.low, prices.p1.low, prices.p2.low); // Pivot Low
    } else {
        SBS = previousState.previousSBS ?? prices.p0.close;
    }

    // Calculate buySellSwitch
    const buySellSwitch = triggerSell ? 1 : (triggerBuy ? 0 : (previousState.buySellSwitch ?? 0));

    // Calculate color signal
    const clrS = (triggerSell && previousState.buySellSwitch === 0) ? 1 :
                 (triggerBuy && previousState.buySellSwitch === 1) ? 0 :
                 (previousState.previousClrS ?? 0);

    // Check if swing size meets minimum requirement
    const swingSize = Math.abs(SBS - prices.p0.close);
    const isValidSwing = swingSize >= previousState.minSwing;

    return {
        buySellSwitch,
        SBS,
        clrS,
        triggerBarHigh,
        triggerBarLow,
        consecutiveHigherCloses,
        consecutiveLowerCloses,
        isNewSell: triggerSell && previousState.buySellSwitch === 0 && isValidSwing,
        isNewBuy: triggerBuy && previousState.buySellSwitch === 1 && isValidSwing
    };
}

function criteriaCheck(backTestFrom, indicators, priceDataObj, priceDateArry, minSwing = 0) {
    const tradeSignals = [];
    let previousState = {
        buySellSwitch: 0,
        previousSBS: null,
        previousClrS: null,
        triggerBarHigh: null,
        triggerBarLow: null,
        consecutiveHigherCloses: 0,
        consecutiveLowerCloses: 0,
        minSwing: minSwing
    };

    for (let i = 5000; i >= 0; i--) {
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
                pivotLevel: signal.SBS,
                consecutiveCloses: signal.isNewBuy ? 
                    signal.consecutiveHigherCloses : 
                    signal.consecutiveLowerCloses
            });
        }

        previousState = {
            ...signal,
            minSwing: minSwing
        };
    }

    return tradeSignals;
}