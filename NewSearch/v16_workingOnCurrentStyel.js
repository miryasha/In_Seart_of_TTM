class TTMScalper {
    constructor() {
        this.trendUp = 1;
        this.trendDown = -1;
        this.digits = 5;
        this.currentTrend = this.trendUp;
        this.lastSignal = null;
    }

    // Main method to analyze current position
    analyzePosition(currentCandles) {
        if (!Array.isArray(currentCandles) || currentCandles.length < 3) {
            console.warn('Need at least 3 candles for analysis');
            return null;
        }

        const signal = this.checkForSignal(currentCandles);
        if (signal) {
            this.currentTrend = signal.type === 'BUY' ? this.trendUp : this.trendDown;
            this.lastSignal = signal;
        }

        return signal;
    }

    checkForSignal(candles) {
        // Check for sell signal if in uptrend
        if (this.currentTrend === this.trendUp) {
            const sellSignal = this.checkSellSignal(candles);
            if (sellSignal) return sellSignal;
        }

        // Check for buy signal if in downtrend
        if (this.currentTrend === this.trendDown) {
            const buySignal = this.checkBuySignal(candles);
            if (buySignal) return buySignal;
        }

        return null;
    }

    checkSellSignal(candles) {
        const swingBar = this.findPivotHigh(candles);
        if (swingBar === -1) return null;

        const current = candles[0];
        const swingBarCandle = candles[swingBar];
        const swingBarPrevCandle = candles[swingBar - 1];

        if (this.safeIsLess(current.close, swingBarPrevCandle.low) && 
            this.safeIsLess(this.findHighestHigh(candles, 0, swingBar), swingBarCandle.high)) {
            
            return {
                type: 'SELL',
                price: swingBarCandle.low,
                trend: this.trendDown
            };
        }

        return null;
    }

    checkBuySignal(candles) {
        const swingBar = this.findPivotLow(candles);
        if (swingBar === -1) return null;

        const current = candles[0];
        const swingBarCandle = candles[swingBar];
        const swingBarPrevCandle = candles[swingBar - 1];

        if (this.safeIsGreater(current.close, swingBarPrevCandle.high) && 
            this.safeIsGreater(this.findLowestLow(candles, 0, swingBar), swingBarCandle.low)) {
            
            return {
                type: 'BUY',
                price: swingBarCandle.high,
                trend: this.trendUp
            };
        }

        return null;
    }

    findPivotHigh(candles) {
        return this.pivot(candles, 0, 'high', candles.length, 2, 2, 1, 1);
    }

    findPivotLow(candles) {
        return this.pivot(candles, 0, 'low', candles.length, 2, 2, 1, -1);
    }

    // Original helper methods remain the same
    pivot(candles, shift, priceType, length, leftStrength, rightStrength, instance, hiLo) {
        if (shift < 0 || shift >= candles.length) return -1;
        length = Math.min(length, candles.length - shift);

        let testPrice;
        let candidatePrice;
        let instanceTest = false;
        let strengthCntr = 0;
        let instanceCntr = 0;
        let lengthCntr = rightStrength;

        while (lengthCntr < length && !instanceTest) {
            if (shift + lengthCntr >= candles.length) break;

            let pivotTest = true;
            const currentCandle = candles[shift + lengthCntr];
            if (!currentCandle || !currentCandle[priceType]) break;

            candidatePrice = currentCandle[priceType];

            // Test left side
            strengthCntr = 1;
            while (pivotTest && (strengthCntr <= leftStrength)) {
                if (shift + lengthCntr + strengthCntr >= candles.length) {
                    pivotTest = false;
                    break;
                }
                testPrice = candles[shift + lengthCntr + strengthCntr][priceType];
                if ((hiLo === 1 && candidatePrice < testPrice) ||
                    (hiLo === -1 && candidatePrice > testPrice)) {
                    pivotTest = false;
                } else {
                    strengthCntr += 1;
                }
            }

            // Test right side
            strengthCntr = 1;
            while (pivotTest && (strengthCntr <= rightStrength)) {
                if (shift + lengthCntr - strengthCntr < 0) {
                    pivotTest = false;
                    break;
                }
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

        return instanceTest ? shift + lengthCntr : -1;
    }

    safeIsLess(first, second) {
        if (first === undefined || second === undefined) return false;
        return this.isLess(first, second);
    }

    safeIsGreater(first, second) {
        if (first === undefined || second === undefined) return false;
        return this.isGreater(first, second);
    }

    isLess(first, second) {
        return Number(first.toFixed(this.digits)) < Number(second.toFixed(this.digits));
    }

    isGreater(first, second) {
        return Number(first.toFixed(this.digits)) > Number(second.toFixed(this.digits));
    }

    findHighestHigh(candles, start, length) {
        let highest = -Infinity;
        const end = Math.min(start + length, candles.length);
        for(let i = start; i < end; i++) {
            if (candles[i] && typeof candles[i].high === 'number' && candles[i].high > highest) {
                highest = candles[i].high;
            }
        }
        return highest;
    }

    findLowestLow(candles, start, length) {
        let lowest = Infinity;
        const end = Math.min(start + length, candles.length);
        for(let i = start; i < end; i++) {
            if (candles[i] && typeof candles[i].low === 'number' && candles[i].low < lowest) {
                lowest = candles[i].low;
            }
        }
        return lowest;
    }
}

module.exports = TTMScalper;



////old one just in case

class TTMScalper {
    constructor() {
        this.trendUp = 1;
        this.trendDown = -1;
        this.digits = 5;
    }

    analyze(candles) {
        if (!Array.isArray(candles) || candles.length < 3) {
            console.warn('Insufficient candle data for analysis');
            return [];
        }

        // Initialize buffers
        const upBuffer = new Array(candles.length).fill(null);
        const downBuffer = new Array(candles.length).fill(null);
        const trendBuffer = new Array(candles.length).fill(this.trendUp);

        // Set initial trend
        trendBuffer[candles.length - 1] = 1;

        // Process each candle
        for(let i = candles.length - 2; i >= 0; i--) {
            if (i < 0 || i >= candles.length) continue;

            upBuffer[i] = null;
            downBuffer[i] = null;
            trendBuffer[i] = trendBuffer[i + 1];

            if (trendBuffer[i] === this.trendUp) {
                let k;
                for (k = 1; (i + k) < candles.length; k++) {
                    if (upBuffer[i + k] !== null && trendBuffer[i + k] === this.trendDown) break;
                }

                k = Math.min(k, candles.length - i - 1);

                const swingBar = this.pivot(candles, i, 'high', k, 2, 2, 1, 1);
                if (swingBar > -1 && swingBar < candles.length - 1) {
                    if (this.safeIsLess(candles[i]?.close, candles[swingBar - 1]?.low) && 
                        this.safeIsLess(this.findHighestHigh(candles, i, swingBar - i), candles[swingBar]?.high)) {
                        
                        // Only set sell signal
                        downBuffer[swingBar] = candles[swingBar].low;
                        trendBuffer[i] = this.trendDown;
                        continue;
                    }
                }
            }

            if (trendBuffer[i] === this.trendDown) {
                let k;
                for (k = 1; (i + k) < candles.length; k++) {
                    if (downBuffer[i + k] !== null && trendBuffer[i + k] === this.trendUp) break;
                }

                k = Math.min(k, candles.length - i - 1);

                const swingBar = this.pivot(candles, i, 'low', k, 2, 2, 1, -1);
                if (swingBar > -1 && swingBar < candles.length - 1) {
                    if (this.safeIsGreater(candles[i]?.close, candles[swingBar - 1]?.high) && 
                        this.safeIsGreater(this.findLowestLow(candles, i, swingBar - i), candles[swingBar]?.low)) {
                        
                        // Only set buy signal
                        upBuffer[swingBar] = candles[swingBar].high;
                        trendBuffer[i] = this.trendUp;
                    }
                }
            }
        }

        return this.generateSignals(upBuffer, downBuffer, trendBuffer);
    }

    pivot(candles, shift, priceType, length, leftStrength, rightStrength, instance, hiLo) {
        // Boundary checks
        if (shift < 0 || shift >= candles.length) return -1;
        length = Math.min(length, candles.length - shift);

        let testPrice;
        let candidatePrice;
        let instanceTest = false;
        let strengthCntr = 0;
        let instanceCntr = 0;
        let lengthCntr = rightStrength;

        while (lengthCntr < length && !instanceTest) {
            // Check if we have valid array access
            if (shift + lengthCntr >= candles.length) break;

            let pivotTest = true;
            const currentCandle = candles[shift + lengthCntr];
            if (!currentCandle || !currentCandle[priceType]) {
                break;
            }

            candidatePrice = currentCandle[priceType];

            // Test left side with bounds checking
            strengthCntr = 1;
            while (pivotTest && (strengthCntr <= leftStrength)) {
                if (shift + lengthCntr + strengthCntr >= candles.length) {
                    pivotTest = false;
                    break;
                }
                testPrice = candles[shift + lengthCntr + strengthCntr][priceType];
                if ((hiLo === 1 && candidatePrice < testPrice) ||
                    (hiLo === -1 && candidatePrice > testPrice)) {
                    pivotTest = false;
                } else {
                    strengthCntr += 1;
                }
            }

            // Test right side with bounds checking
            strengthCntr = 1;
            while (pivotTest && (strengthCntr <= rightStrength)) {
                if (shift + lengthCntr - strengthCntr < 0) {
                    pivotTest = false;
                    break;
                }
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

        return instanceTest ? shift + lengthCntr : -1;
    }

    safeIsLess(first, second) {
        if (first === undefined || second === undefined) return false;
        return this.isLess(first, second);
    }

    safeIsGreater(first, second) {
        if (first === undefined || second === undefined) return false;
        return this.isGreater(first, second);
    }

    isLess(first, second) {
        return Number(first.toFixed(this.digits)) < Number(second.toFixed(this.digits));
    }

    isGreater(first, second) {
        return Number(first.toFixed(this.digits)) > Number(second.toFixed(this.digits));
    }

    findHighestHigh(candles, start, length) {
        let highest = -Infinity;
        const end = Math.min(start + length, candles.length);
        for(let i = start; i < end; i++) {
            if (candles[i] && typeof candles[i].high === 'number' && candles[i].high > highest) {
                highest = candles[i].high;
            }
        }
        return highest;
    }

    findLowestLow(candles, start, length) {
        let lowest = Infinity;
        const end = Math.min(start + length, candles.length);
        for(let i = start; i < end; i++) {
            if (candles[i] && typeof candles[i].low === 'number' && candles[i].low < lowest) {
                lowest = candles[i].low;
            }
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

module.exports = TTMScalper;