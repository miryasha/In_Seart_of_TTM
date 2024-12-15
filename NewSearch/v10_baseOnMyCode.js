class TTMScalper {
    constructor() {
        this.trendUp = 1;
        this.trendDown = -1;
        this.digits = 5; // Default precision, adjust as needed
    }

    // Core analysis function
    analyze(candles) {
        // Initialize buffers
        const upBuffer = new Array(candles.length).fill(null);
        const downBuffer = new Array(candles.length).fill(null);
        const trendBuffer = new Array(candles.length).fill(this.trendUp);

        // Process each candle
        for(let i = candles.length - 2; i >= 0; i--) {
            if (trendBuffer[i] === this.trendUp) {
                // Find last down signal
                let k;
                for (k = 1; (i + k) < candles.length; k++) {
                    if (downBuffer[i + k] !== null && trendBuffer[i + k] === this.trendDown) break;
                }

                // Check for swing high
                const swingBar = this.findSwingHigh(candles, i, 2, k);
                if (swingBar > -1) {
                    if (this.isLess(candles[i].close, candles[swingBar - 1].low) && 
                        this.isLess(this.findHighestHigh(candles, i, swingBar - i), candles[swingBar].high)) {
                        
                        upBuffer[swingBar] = candles[swingBar].high;
                        downBuffer[swingBar] = candles[swingBar].low;
                        trendBuffer[i] = this.trendDown;
                        continue;
                    }
                }
            }

            if (trendBuffer[i] === this.trendDown) {
                // Find last up signal
                let k;
                for (k = 1; (i + k) < candles.length; k++) {
                    if (upBuffer[i + k] !== null && trendBuffer[i + k] === this.trendUp) break;
                }

                // Check for swing low
                const swingBar = this.findSwingLow(candles, i, 2, k);
                if (swingBar > -1) {
                    if (this.isGreater(candles[i].close, candles[swingBar - 1].high) && 
                        this.isGreater(this.findLowestLow(candles, i, swingBar - i), candles[swingBar].low)) {
                        
                        downBuffer[swingBar] = candles[swingBar].high;
                        upBuffer[swingBar] = candles[swingBar].low;
                        trendBuffer[i] = this.trendUp;
                    }
                }
            }
        }

        return this.generateSignals(upBuffer, downBuffer, trendBuffer);
    }

    // Helper methods
    isLess(first, second) {
        return Number(first.toFixed(this.digits)) < Number(second.toFixed(this.digits));
    }

    isGreater(first, second) {
        return Number(first.toFixed(this.digits)) > Number(second.toFixed(this.digits));
    }

    findSwingHigh(candles, shift, strength, length) {
        let highestBar = -1;
        let highestPrice = -Infinity;

        for(let i = shift; i < Math.min(shift + length, candles.length); i++) {
            let isSwingHigh = true;
            
            // Check left side
            for(let j = 1; j <= strength && i + j < candles.length; j++) {
                if (candles[i].high <= candles[i + j].high) {
                    isSwingHigh = false;
                    break;
                }
            }
            
            // Check right side
            for(let j = 1; j <= strength && i - j >= 0; j++) {
                if (candles[i].high <= candles[i - j].high) {
                    isSwingHigh = false;
                    break;
                }
            }

            if (isSwingHigh && candles[i].high > highestPrice) {
                highestPrice = candles[i].high;
                highestBar = i;
            }
        }

        return highestBar;
    }

    findSwingLow(candles, shift, strength, length) {
        let lowestBar = -1;
        let lowestPrice = Infinity;

        for(let i = shift; i < Math.min(shift + length, candles.length); i++) {
            let isSwingLow = true;
            
            // Check left side
            for(let j = 1; j <= strength && i + j < candles.length; j++) {
                if (candles[i].low >= candles[i + j].low) {
                    isSwingLow = false;
                    break;
                }
            }
            
            // Check right side
            for(let j = 1; j <= strength && i - j >= 0; j++) {
                if (candles[i].low >= candles[i - j].low) {
                    isSwingLow = false;
                    break;
                }
            }

            if (isSwingLow && candles[i].low < lowestPrice) {
                lowestPrice = candles[i].low;
                lowestBar = i;
            }
        }

        return lowestBar;
    }

    findHighestHigh(candles, start, length) {
        let highest = -Infinity;
        for(let i = start; i < Math.min(start + length, candles.length); i++) {
            if (candles[i].high > highest) highest = candles[i].high;
        }
        return highest;
    }

    findLowestLow(candles, start, length) {
        let lowest = Infinity;
        for(let i = start; i < Math.min(start + length, candles.length); i++) {
            if (candles[i].low < lowest) lowest = candles[i].low;
        }
        return lowest;
    }

    generateSignals(upBuffer, downBuffer, trendBuffer) {
        const signals = [];
        
        for(let i = 0; i < upBuffer.length; i++) {
            if (upBuffer[i] !== null) {
                signals.push({
                    index: i,
                    type: 'BUY',
                    price: upBuffer[i],
                    trend: trendBuffer[i]
                });
            }
            if (downBuffer[i] !== null) {
                signals.push({
                    index: i,
                    type: 'SELL',
                    price: downBuffer[i],
                    trend: trendBuffer[i]
                });
            }
        }
        
        return signals;
    }
}

// Example usage
const ttmScalper = new TTMScalper();

// Example candle data format
const exampleCandles = [
    { open: 100, high: 105, low: 98, close: 103 },
    { open: 103, high: 107, low: 102, close: 106 },
    // ... more candles
];

const signals = ttmScalper.analyze(exampleCandles);

export default TTMScalper;