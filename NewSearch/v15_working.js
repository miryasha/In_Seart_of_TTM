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







































// const express = require('express');
// const router = express.Router();
// const getPrice = require('../../services/api/getPrice');
// const getIndicators = require("../../services/api/TTMSclpterIndicatior");
// const TTMScalper = require('./TTMScalper');

// function formatCandles(priceDataObj, priceDateArray) {
//     const sortedDates = [...priceDateArray].sort((a, b) => new Date(a) - new Date(b));
    
//     return sortedDates.map(date => ({
//         open: parseFloat(priceDataObj[date]["1. open"]),
//         high: parseFloat(priceDataObj[date]["2. high"]),
//         low: parseFloat(priceDataObj[date]["3. low"]),
//         close: parseFloat(priceDataObj[date]["4. close"]),
//         volume: parseFloat(priceDataObj[date]["5. volume"]),
//         date: date
//     }));
// }

// function criteriaCheck(backTestFrom, indicators, priceDataObj, priceDateArray) {
//     try {
//         const ttmScalper = new TTMScalper();
//         const candles = formatCandles(priceDataObj, priceDateArray);
//         const signals = ttmScalper.analyze(candles);
        
//         // Convert signals to simple format
//         const simplifiedSignals = signals.map(signal => ({
//             type: signal.type,
//             date: candles[signal.index].date
//         }));
        
//         return {
//             signals: simplifiedSignals,
//             currentSignal: simplifiedSignals.length > 0 ? simplifiedSignals[simplifiedSignals.length - 1] : null
//         };
        
//     } catch (error) {
//         console.error('Error in criteriaCheck:', error);
//         throw new Error(`Failed to process TTM Scalper analysis: ${error.message}`);
//     }
// }

// router.post('/', async (req, res) => {
//     try {
//         const { ticker, marketName, timeFrame } = req.body;

//         // Validate timeFrame
//         const validTimeFrames = ['daily', 'weekly', 'monthly'];
//         const normalizedTimeFrame = timeFrame.toLowerCase();
//         if (!validTimeFrames.includes(normalizedTimeFrame)) {
//             throw new Error(`Invalid timeFrame: ${timeFrame}. Must be one of: ${validTimeFrames.join(', ')}`);
//         }

//         const pricesInfo = await getPrice(ticker, marketName, normalizedTimeFrame);
//         const indicators = await getIndicators(ticker, timeFrame);
//         const backTestFrom = pricesInfo.priceDateArry.length - 1;

//         const results = criteriaCheck(
//             backTestFrom, 
//             indicators, 
//             pricesInfo.priceDataObj, 
//             pricesInfo.priceDateArry
//         );

//         res.json({
//             ticker,
//             timeFrame: normalizedTimeFrame,
//             signals: results.signals,
//             currentSignal: results.currentSignal
//         });

//     } catch (error) {
//         console.error('Route handler error:', error);
//         res.status(500).json({ error: error.message });
//     }
// });

// module.exports = router;