
function detectPriceSignal(prices, previousState = {
    buySellSwitch: null,
    previousSBS: null,
    previousClrS: null
}) {
    // Calculate triggers using close prices
    const triggerSell = (prices.p1.close < prices.p0.close) &&
        (prices.p2.close < prices.p1.close || prices.p3.close < prices.p1.close) ? 1 : 0;

    const triggerBuy = (prices.p1.close > prices.p0.close) &&
        (prices.p2.close > prices.p1.close || prices.p3.close > prices.p1.close) ? 1 : 0;

    // Calculate buySellSwitch
    let buySellSwitch;
    if (triggerSell) {
        buySellSwitch = 1;
    } else if (triggerBuy) {
        buySellSwitch = 0;
    } else {
        buySellSwitch = previousState.buySellSwitch !== null ? previousState.buySellSwitch : 0;
    }

    // Calculate SBS (Support/Resistance level)
    let SBS;
    if (triggerSell && previousState.buySellSwitch === false) {
        SBS = prices.p0.high;
    } else if (triggerBuy && previousState.buySellSwitch) {
        SBS = prices.p0.low;
    } else {
        SBS = previousState.previousSBS !== null ? previousState.previousSBS : prices.p0.close;
    }

    // Calculate color signal
    let clrS;
    if (triggerSell && previousState.buySellSwitch === false) {
        clrS = 1; // Red - Bear Signal
    } else if (triggerBuy && previousState.buySellSwitch) {
        clrS = 0; // Green - Bull Signal
    } else {
        clrS = previousState.previousClrS !== null ? previousState.previousClrS : 0;
    }

    return {
        buySellSwitch,
        SBS,
        clrS,
        isBearish: clrS === 1,
        isBullish: clrS === 0
    };
}

