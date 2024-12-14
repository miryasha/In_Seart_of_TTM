function detectPriceSignal(prices, previousState = {
    buySellSwitch: null,
    previousSBS: null,
    previousClrS: null
}) {
    // Calculate triggers using close prices - NOTE THE REVERSED COMPARISON
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
    if (triggerSell && previousState.buySellSwitch === 0) { // Changed from false to 0
        SBS = prices.p0.high;
    } else if (triggerBuy && previousState.buySellSwitch === 1) { // Changed from truthy to explicit 1
        SBS = prices.p0.low;
    } else {
        SBS = previousState.previousSBS !== null ? previousState.previousSBS : prices.p0.close;
    }

    // Calculate color signal
    let clrS;
    if (triggerSell && previousState.buySellSwitch === 0) { // Changed from false to 0
        clrS = 1; // Red - Bear Signal
    } else if (triggerBuy && previousState.buySellSwitch === 1) { // Changed from truthy to explicit 1
        clrS = 0; // Green - Bull Signal
    } else {
        clrS = previousState.previousClrS !== null ? previousState.previousClrS : 0;
    }

    return {
        buySellSwitch,
        SBS,
        clrS,
        isBearish: clrS === 1,
        isBullish: clrS === 0,
        triggerBuy,
        triggerSell
    };
}




// // Log trade signals
// if (priceSignal.triggerBuy && previousState.buySellSwitch === 1) {
//     console.log(`BUY Signal on ${date} at price ${prices.p0.close}`);
//     tradeInfoBull.push({
//         signalDate: date,
//         signalOpeningPrice: prices.p0.open,
//         signalHighPrice: prices.p0.high,
//         signalLowPrice: prices.p0.low,
//         signalClosingPrice: prices.p0.close,
//         direction: 'Bull',
//         supportLevel: priceSignal.SBS
//     });
// }

// if (priceSignal.triggerSell && previousState.buySellSwitch === 0) {
//     console.log(`SELL Signal on ${date} at price ${prices.p0.close}`);
//     tradeInfoBear.push({
//         signalDate: date,
//         signalOpeningPrice: prices.p0.open,
//         signalHighPrice: prices.p0.high,
//         signalLowPrice: prices.p0.low,
//         signalClosingPrice: prices.p0.close,
//         direction: 'Bear',
//         resistanceLevel: priceSignal.SBS
//     });
// }