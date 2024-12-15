class TTMScalper {
    constructor() {
        this.trendUp = 1;
        this.trendDown = -1;
        this.digits = 5; // Default precision
    }

    analyze(candles) {
        // Initialize buffers
        const upBuffer = new Array(candles.length).fill(null);
        const downBuffer = new Array(candles.length).fill(null);
        const trendBuffer = new Array(candles.length).fill(this.trendUp);

        // Set initial trend (matches original code's behavior)
        trendBuffer[candles.length - 1] = 1;

        // Process each candle
        for(let i = candles.length - 2; i >= 0; i--) {
            // Initialize buffers for current index
            upBuffer[i] = null;
            downBuffer[i] = null;
            trendBuffer[i] = trendBuffer[i + 1];  // Inherit trend from previous bar

            if (trendBuffer[i] === this.trendUp) {
                // Find last down signal (exact match to original logic)
                let k;
                for (k = 1; (i + k) < candles.length; k++) {
                    if (upBuffer[i + k] !== null && trendBuffer[i + k] === this.trendDown) break;
                }

                const swingBar = this.pivot(candles, i, 'high', k, 2, 2, 1, 1);
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
                // Find last up signal (exact match to original logic)
                let k;
                for (k = 1; (i + k) < candles.length; k++) {
                    if (downBuffer[i + k] !== null && trendBuffer[i + k] === this.trendUp) break;
                }

                const swingBar = this.pivot(candles, i, 'low', k, 2, 2, 1, -1);
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

    // Exact implementation of the original pivot logic
    pivot(candles, shift, priceType, length, leftStrength, rightStrength, instance, hiLo) {
        let testPrice;
        let candidatePrice;
        let instanceTest = false;
        let strengthCntr = 0;
        let instanceCntr = 0;
        let lengthCntr = rightStrength;

        while (lengthCntr < length && !instanceTest) {
            let pivotTest = true;
            candidatePrice = candles[shift + lengthCntr][priceType];

            // Test left side (matches original exactly)
            strengthCntr = 1;
            while (pivotTest && (strengthCntr <= leftStrength)) {
                testPrice = candles[shift + lengthCntr + strengthCntr][priceType];
                if ((hiLo === 1 && candidatePrice < testPrice) ||
                    (hiLo === -1 && candidatePrice > testPrice)) {
                    pivotTest = false;
                } else {
                    strengthCntr += 1;
                }
            }

            // Test right side (matches original exactly)
            strengthCntr = 1;
            while (pivotTest && (strengthCntr <= rightStrength)) {
                testPrice = candles[shift + lengthCntr - strengthCntr][priceType];
                if ((hiLo === 1 && candidatePrice <= testPrice) ||
                    (hiLo === -1 && candidatePrice >= testPrice)) {
                    pivotTest = false;
                } else {
                    strengthCntr += 1;
                }
            }

            if (pivotTest) instanceCntr += 1;
            if (instanceCntr === instance) {
                instanceTest = true;
            } else {
                lengthCntr += 1;
            }
        }

        if (instanceTest) {
            return shift + lengthCntr;
        } else {
            return -1;
        }
    }

    isLess(first, second) {
        return Number(first.toFixed(this.digits)) < Number(second.toFixed(this.digits));
    }

    isGreater(first, second) {
        return Number(first.toFixed(this.digits)) > Number(second.toFixed(this.digits));
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

// Changed from ES modules export to CommonJS module.exports
module.exports = TTMScalper;