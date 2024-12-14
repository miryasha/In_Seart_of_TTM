class TTMScalper {
    constructor(width = 2) {
        this.width = width;
        this.previousBuySellSwitch = null;
        this.previousSBS = null;
        this.previousClrS = null;
    }

    calculate(data) {
        if (!Array.isArray(data) || data.length < 4) {
            throw new Error('Need at least 4 bars of price data');
        }

        const signals = [];
        const results = [];

        // Process each bar
        for (let i = 3; i < data.length; i++) {
            const close = data[i].close;
            const close1 = data[i-1].close;
            const close2 = data[i-2].close;
            const close3 = data[i-3].close;
            const high = data[i].high;
            const low = data[i].low;

            // Calculate triggers
            const triggerSell = (close1 < close) && (close2 < close1 || close3 < close1) ? 1 : 0;
            const triggerBuy = (close1 > close) && (close2 > close1 || close3 > close1) ? 1 : 0;

            // Calculate buySellSwitch
            let buySellSwitch;
            if (triggerSell) {
                buySellSwitch = 1;
            } else if (triggerBuy) {
                buySellSwitch = 0;
            } else {
                buySellSwitch = this.previousBuySellSwitch !== null ? this.previousBuySellSwitch : 0;
            }

            // Calculate SBS (Support/Resistance level)
            let SBS;
            if (triggerSell && this.previousBuySellSwitch === false) {
                SBS = high;
            } else if (triggerBuy && this.previousBuySellSwitch) {
                SBS = low;
            } else {
                SBS = this.previousSBS !== null ? this.previousSBS : close;
            }

            // Calculate color signal
            let clrS;
            if (triggerSell && this.previousBuySellSwitch === false) {
                clrS = 1; // Red
            } else if (triggerBuy && this.previousBuySellSwitch) {
                clrS = 0; // Green
            } else {
                clrS = this.previousClrS !== null ? this.previousClrS : 0;
            }

            // Store results
            const result = {
                timestamp: data[i].timestamp || i,
                close: close,
                SBS: SBS,
                signal: clrS === 0 ? 'buy' : 'sell',
                value: SBS,
                width: this.width
            };

            // Update previous values for next iteration
            this.previousBuySellSwitch = buySellSwitch;
            this.previousSBS = SBS;
            this.previousClrS = clrS;

            results.push(result);
        }

        return results;
    }
}

// Example usage:
const sampleData = [
    { timestamp: "2024-01-01", open: 100, high: 105, low: 98, close: 103 },
    { timestamp: "2024-01-02", open: 103, high: 107, low: 101, close: 104 },
    { timestamp: "2024-01-03", open: 104, high: 109, low: 102, close: 103 },
    { timestamp: "2024-01-04", open: 103, high: 106, low: 100, close: 105 },
    { timestamp: "2024-01-05", open: 105, high: 108, low: 103, close: 106 }
];

try {
    const ttmScalper = new TTMScalper(2); // width = 2
    const signals = ttmScalper.calculate(sampleData);
    console.log('TTM Scalper Signals:', JSON.stringify(signals, null, 2));
} catch (error) {
    console.error('Error calculating TTM Scalper signals:', error.message);
}

module.exports = TTMScalper;