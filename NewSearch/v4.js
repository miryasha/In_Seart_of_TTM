function detectPriceSignal(prices, previousState = {
    buySellSwitch: null,
    previousSBS: null,
    previousClrS: null
}) {
    // Exactly matching the original logic:
    // triggerSell = iff(iff(close[1] < close,1,0) and (close[2] < close[1] or close[3] <close[1]),1,0)
    const triggerSell = (prices.p1.close < prices.p0.close) && 
                       (prices.p2.close < prices.p1.close || prices.p3.close < prices.p1.close) ? 1 : 0;

    // triggerBuy = iff(iff(close[1] > close,1,0) and (close[2] > close[1] or close[3] > close[1]),1,0)
    const triggerBuy = (prices.p1.close > prices.p0.close) && 
                      (prices.p2.close > prices.p1.close || prices.p3.close > prices.p1.close) ? 1 : 0;

    // buySellSwitch = iff(triggerSell, 1, iff(triggerBuy, 0, nz(buySellSwitch[1])))
    const buySellSwitch = triggerSell ? 1 : (triggerBuy ? 0 : (previousState.buySellSwitch ?? 0));

    // SBS = iff(triggerSell and buySellSwitch[1] == false, high, iff(triggerBuy and buySellSwitch[1], low, nz(SBS[1])))
    const SBS = (triggerSell && previousState.buySellSwitch === 0) ? prices.p0.high : 
                (triggerBuy && previousState.buySellSwitch === 1) ? prices.p0.low : 
                (previousState.previousSBS ?? prices.p0.close);

    // clr_s = iff(triggerSell and buySellSwitch[1] == false, 1, iff(triggerBuy and buySellSwitch[1], 0, nz(clr_s[1])))
    const clrS = (triggerSell && previousState.buySellSwitch === 0) ? 1 :
                 (triggerBuy && previousState.buySellSwitch === 1) ? 0 :
                 (previousState.previousClrS ?? 0);

    return {
        buySellSwitch,
        SBS,
        clrS,
        isNewSell: triggerSell && previousState.buySellSwitch === 0,
        isNewBuy: triggerBuy && previousState.buySellSwitch === 1
    };
}

function criteriaCheck(backTestFrom, indicators, priceDataObj, priceDateArry) {
    const tradeSignals = [];
    let previousState = {
        buySellSwitch: 0,  // Initialize to 0 as per original script
        previousSBS: null,
        previousClrS: null
    };

    for (let i = 5000; i >= 0; i--) {
        const prices = {
            p0: {
                close: parseFloat(priceDataObj[priceDateArry[i]]["4. close"]),
                high: parseFloat(priceDataObj[priceDateArry[i]]["2. high"]),
                low: parseFloat(priceDataObj[priceDateArry[i]]["3. low"])
            },
            p1: {
                close: parseFloat(priceDataObj[priceDateArry[i + 1]]["4. close"])
            },
            p2: {
                close: parseFloat(priceDataObj[priceDateArry[i + 2]]["4. close"])
            },
            p3: {
                close: parseFloat(priceDataObj[priceDateArry[i + 3]]["4. close"])
            }
        };

        const signal = detectPriceSignal(prices, previousState);

        // Record new signals with their dates
        if (signal.isNewBuy) {
            console.log(`BUY Signal on ${priceDateArry[i]} at price ${prices.p0.close}`);
            tradeSignals.push({
                date: priceDateArry[i],
                type: 'BUY',
                price: prices.p0.close,
                SBS: signal.SBS
            });
        }
        if (signal.isNewSell) {
            console.log(`SELL Signal on ${priceDateArry[i]} at price ${prices.p0.close}`);
            tradeSignals.push({
                date: priceDateArry[i],
                type: 'SELL',
                price: prices.p0.close,
                SBS: signal.SBS
            });
        }

        // Update state for next iteration
        previousState = {
            buySellSwitch: signal.buySellSwitch,
            previousSBS: signal.SBS,
            previousClrS: signal.clrS
        };
    }

    return tradeSignals;
}