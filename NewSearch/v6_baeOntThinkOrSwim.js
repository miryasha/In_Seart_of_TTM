function detectScalperSignals(prices, previousState = {
    previousCloses: [],
    minSwing: 0
}) {
    // Initialize state if needed
    const closes = [...(previousState.previousCloses || []), prices.p0.close];
    if (closes.length > 3) closes.shift(); // Keep only last 3 closes

    // Check for 3 consecutive lower closes (Sell Signal)
    const isThreeLowerCloses = closes.length === 3 && 
                              closes[2] < closes[1] && 
                              closes[1] < closes[0];

    // Check for 3 consecutive higher closes (Buy Signal)
    const isThreeHigherCloses = closes.length === 3 && 
                               closes[2] > closes[1] && 
                               closes[1] > closes[0];

    // Calculate swing size
    const swingSize = Math.abs(prices.p0.high - prices.p0.low);
    const isValidSwing = swingSize >= previousState.minSwing;

    // Determine signals
    const sellSignal = isThreeLowerCloses && isValidSwing;
    const buySignal = isThreeHigherCloses && isValidSwing;

    return {
        previousCloses: closes,
        sellSignal,
        buySignal,
        pivotHigh: sellSignal ? prices.p0.high : null,
        pivotLow: buySignal ? prices.p0.low : null
    };
}

function criteriaCheck(backTestFrom, indicators, priceDataObj, priceDateArry, minSwing = 0) {
    const signals = [];
    let previousState = {
        previousCloses: [],
        minSwing
    };

    for (let i = 5000; i >= 3; i--) {
        const prices = {
            p0: {
                close: parseFloat(priceDataObj[priceDateArry[i]]["4. close"]),
                high: parseFloat(priceDataObj[priceDateArry[i]]["2. high"]),
                low: parseFloat(priceDataObj[priceDateArry[i]]["3. low"])
            }
        };

        const result = detectScalperSignals(prices, previousState);

        // Record signals
        if (result.sellSignal || result.buySignal) {
            signals.push({
                date: priceDateArry[i],
                type: result.sellSignal ? 'SELL' : 'BUY',
                price: prices.p0.close,
                pivotLevel: result.sellSignal ? result.pivotHigh : result.pivotLow,
                swingSize: Math.abs(prices.p0.high - prices.p0.low)
            });

            console.log(`${result.sellSignal ? 'SELL' : 'BUY'} Signal on ${priceDateArry[i]} at ${prices.p0.close}`);
        }

        previousState = {
            previousCloses: result.previousCloses,
            minSwing
        };
    }

    return signals;
}


// // minSwing parameter can be adjusted based on the instrument and timeframe
// const signals = criteriaCheck(backTestFrom, indicators, priceDataObj, priceDateArry, 1.0);